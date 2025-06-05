from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tiktoken
import re
import json
from datetime import datetime
import ollama
from typing import List, Dict, Any, Optional
import os
from asgiref.wsgi import WsgiToAsgi
from history_manager_help import HistoryManager
from functools import lru_cache

# -------------------------------------------------------------------
# 1. Müşteri Hizmetleri için Optimize Edilmiş Ayarlar
# -------------------------------------------------------------------

# GPU kullanımını zorla
os.environ["OLLAMA_FORCE_CUDA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Flask app // CORS // ASGI
app = Flask(__name__)
CORS(app)
asgi_app = WsgiToAsgi(app)

# Ollama client - Model - Sıcaklık (müşteri hizmetleri için düşük)
client = ollama.Client()
MODEL_NAME = "quikHelper:latest"
DEFAULT_TEMPERATURE = 1.0  # Daha tutarlı yanıtlar için düşük

# Müşteri hizmetleri için kısa context limitleri
TOTAL_MAX_TOKENS: int = 32768  # 32k yeterli müşteri hizmetleri için
RESERVED_OUTPUT_TOKENS: int = 8192  # Kısa yanıtlar için yeterli
AVAILABLE_INPUT_TOKENS: int = TOTAL_MAX_TOKENS - RESERVED_OUTPUT_TOKENS

# HistoryManager - Müşteri hizmetleri optimizasyonu
history_manager = HistoryManager(
    total_max_tokens=TOTAL_MAX_TOKENS,
    reserved_output_tokens=RESERVED_OUTPUT_TOKENS,
    client=client,
    model_name=MODEL_NAME,
    max_conversation_turns=6,  # Son 6 mesaj yeterli
    aggressive_summarize=True   # Agresif özetleme
)

# tiktoken için
tokenizer = tiktoken.get_encoding("cl100k_base")

# -------------------------------------------------------------------
# 2. Hızlı Yardımcı Fonksiyonlar
# -------------------------------------------------------------------

@lru_cache(maxsize=8_192)  # Cache boyutu küçültüldü
def count_tokens(text: str) -> int:
    """Hızlı token sayımı için cache"""
    return len(tokenizer.encode(text))

def quick_summarize(text: str, max_words: int = 200) -> str:
    """Müşteri hizmetleri için hızlı özetleme"""
    words = text.split()
    if len(words) <= max_words:
        return text
    # İlk ve son kısmı al, ortayı özetle
    start_words = words[:max_words//3]
    end_words = words[-max_words//3:]
    return " ".join(start_words) + " ... " + " ".join(end_words)

def append_current_date(query: str) -> str:
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    return f"{query} {current_date}"

# -------------------------------------------------------------------
# 3. Müşteri Hizmetleri için Kısa ve Net Sistem Promptu
# -------------------------------------------------------------------

CUSTOMER_SERVICE_PROMPT = """Senin adın 'quik'. SGDD-ASAM tarafından geliştirilen bir yapay zeka asistanısın. Görevin, SGDD-ASAM'ın 'quik-chatbot' platformu için kullanıcılara hızlı ve etkili müşteri desteği sağlamaktır. Kullanıcılara her zaman Türkçe olarak yardımcı olmalı ve soruları çözene kadar destek sunmalısın. Yardım ettiğin platformun adı 'quik-chatbot' ve bu platform dışında başka konularla ilgilenmemen beklenmektedir.

## HIZLI ÇÖZÜMLER:

**TEMA/GÖRÜNÜM:** Ayarlar → Genel → Görünüm (Aydınlık/Karanlık/Sistem seçenekleri var)
**WEB ARAMA:** Ayarlar → Genel → Web Arama (Açık/Kapalı/Devre Dışı seçenekleri var)
**DİL:** Ayarlar → Genel → Dil Seçimi (TR/EN/AR seçenekleri var)
**BİLDİRİMLER:** Ayarlar → Bildirimler → Bildirim Ayarları (Hepsini açmasını öner)
**KİŞİSELLEŞTİRME:** Ayarlar → Kişiselleştirme → Özel Talimatlar (Kendi talimatını oluşturabilir veya 3 farklı talimattan birini seçebilir)
**SESLİ KONUŞMA:** Ayarlar → Konuşma → Asistan Seçimi (4 farklı karakter seçeneği var) + Mikrofon Testi yapabilir
**SOHBET LİMİTİ (Yeni Sohbet oluşturamıyorsa):** Ayarlar → Geçmiş (200/200 ise tüm sohbeti arşivleyebilir veya silebilir)
**HESAP BAĞLAMA:** Ayarlar → Bağlı Uygulamalar (Google: Gmail/Drive/Chat/Takvim, Microsoft: OneDrive/Teams uygulamalarını bağlayabilir)
**GÜVENLİK:** Ayarlar → Hesap → Güvenlik → İki Aşamalı Doğrulama
**ŞİFRE:** Ayarlar → Hesap → Şifre (Şifremi Unuttum/Şifremi Sıfırla/Şifremi Değiştir seçenekleri var)
**GİZLİLİK POLİTİKASI:** Gizlilik politikası oldukça kapsamlı. Link: chat.quikai.com/privacy-policy
**KULLANIM ŞARTLARI:** Kullanım şartları oldukça kapsamlı. Link: chat.quikai.com/terms-of-services
**ÜYE OLMAK:** SGDD-ASAM Kullanıcı Adı ve Şifresi, Google hesabı, Microsoft hesabı ile giriş yapılabilir veya yeni hesap oluşturulabilir. Yeni hesap oluşturmak için link: register.quikai.com

**SGDD-ASAM SİSTEMLERİNE BAĞLANTI SORUNLARI (BIODATA/DGBYS/EMTIA/MEYER):**
1. SGDD-ASAM kullanıcı adı/şifre kontrol
2. VPN kontrol
3. Ayarlar → Bağlı Sistemler kontrol
4. Çözülmezse IT Birimi

**MsSQL/PYTHON SORUNLARI:**
1. Ayarlar → Bağlı Programlar kontrol (Açık mı kapalı mı)
2. Parametre kontrol
3. VPN kontrol
4. Server bağlantısı (17/34 sonlu) kontrol
5. Çözülmezse IT Birimi

**KURALLAR:**
- Sadece quik-chatbot desteği
- Hızlı, kısa, net yanıtlar
- Sorun çözülene kadar takip et"""

# -------------------------------------------------------------------
# 4. Hızlı Chat Endpoint
# -------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
async def chat() -> Response:
    try:
        req_data: Dict[str, Any] = request.get_json()
        user_message: str = req_data.get("message", "")
        history: List[Dict[str, str]] = req_data.get("history", [])
        temperature: float = req_data.get("temperature", DEFAULT_TEMPERATURE)

        if not user_message:
            return jsonify({"reply": "Mesaj sağlanmadı"}), 400

        # Geçmişi hızlı formatla
        history = [
            {"role": msg["sender"], "text": msg["text"]}
            for msg in history[-8:]  # Son 8 mesaj yeterli müşteri hizmetleri için
            if msg.get("sender") in ["user", "assistant"] and msg.get("text")
        ]

        # Hızlı özetleme
        user_message = quick_summarize(user_message, max_words=150)

        # Hızlı geçmiş işleme
        processed_history, _ = await history_manager.process_history(
            history,
            user_message,
            force_summarize=(len(history) > 4)  # 4 mesajdan sonra özetle
        )

        # Hızlı mesaj formatı
        messages = [{"role": "system", "content": CUSTOMER_SERVICE_PROMPT}]
        messages += [{"role": h["role"], "content": h["text"]} for h in processed_history[-4:]]  # Son 4 mesaj
        messages.append({"role": "user", "content": user_message})

        # Hızlı streaming
        def generate() -> str:
            try:
                stream = client.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "top_k": 32,  # Daha hızlı için düşük
                        "top_p": 0.85,  # Daha odaklı yanıtlar
                        "repeat_penalty": 1.2,  # Hafif tekrar önleme
                        "max_tokens": 512,  # Kısa yanıtlar için yeterli
                        "stop": ["<end_of_turn>", "\n\n\n"]  # Fazla uzamasın
                    }
                )

                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield f"data: {json.dumps({'text': chunk['message']['content']})}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                yield f"data: error: {str(e)}\n\n"

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["X-Accel-Buffering"] = "no"
        return response

    except Exception as e:
        return jsonify({"reply": f"Hata: {str(e)}"}), 500

# -------------------------------------------------------------------
# 5. Sağlık Kontrolü Endpoint'i
# -------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Hızlı sağlık kontrolü"""
    return jsonify({"status": "OK", "service": "quik-customer-service"}), 200

if __name__ == '__main__':
    app.run(debug=False, threaded=True)  # Production için debug=False