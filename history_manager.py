from typing import List, Dict, Any, Tuple, Optional, Set
import tiktoken
from functools import lru_cache
import asyncio
from rapidfuzz import fuzz
from datetime import datetime, timedelta
import re
import json
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt

# -------------------------------------------------------------------
# 1. Gelişmiş Token ve Bellek Yönetimi
# -------------------------------------------------------------------

tokenizer = tiktoken.get_encoding("cl100k_base")

@lru_cache(maxsize=16_384)
def count_tokens(text: str) -> int:
    """LRU-cache ile aynı stringler için tekrar hesaplama yapmayı önler."""
    return len(tokenizer.encode(text))

class HistoryManager:
    def __init__(self, 
                 total_max_tokens: int = 131072, 
                 reserved_output_tokens: int = 65536,
                 client = None,
                 model_name: str = "Quik_v2:latest"):
        self.TOTAL_MAX_TOKENS = total_max_tokens
        self.RESERVED_OUTPUT_TOKENS = reserved_output_tokens
        self.AVAILABLE_INPUT_TOKENS = self.TOTAL_MAX_TOKENS - self.RESERVED_OUTPUT_TOKENS
        self.client = client  # Ollama client
        self.model_name = model_name
        self.conversation_state = {}  # Sohbet durumu metadata tutma
        self.key_concepts = set()  # Önemli kavramlar için set
        self.entity_memory = {}  # Varlık hafızası
        self.session_start_time = datetime.now()
        
    # -------------------------------------------------------------------
    # 2. Akıllı Geçmiş Trimming Sistemi
    # -------------------------------------------------------------------
    
    def trim_history_by_tokens(self, 
                             history: List[Dict[str, str]],
                             max_tokens: int = None,
                             decay_factor: float = 0.9) -> List[Dict[str, str]]:
        """
        Gelişmiş history trimming:
        - Token bütçesine göre adapt olur
        - Eski mesajlara göre azalan bir önem faktörü uygular
        - Önemli mesajları korur (etiketli ya da yüksek içerik değeri)
        """
        if max_tokens is None:
            max_tokens = self.AVAILABLE_INPUT_TOKENS
            
        if not history:
            return []
            
        # Sohbet boyunca önemli mesajları belirle
        important_indices = self._identify_important_messages(history)
        
        # Mesajların ağırlıklarını hesapla
        weights = []
        for i, msg in enumerate(history):
            # Baz ağırlık
            weight = 1.0
            # Önemli mesajsa daha yüksek
            if i in important_indices:
                weight *= 2.0
            # Mesaj ne kadar eskiyse ağırlığı o kadar az - geometrik decay
            position_factor = decay_factor ** (len(history) - i - 1)
            weight *= position_factor
            weights.append(weight)
            
        # Mesajların token uzunluklarını hesapla
        token_counts = []
        for msg in history:
            tk = msg.get('_tk')
            if tk is None:
                tk = count_tokens(msg.get("text", ""))
            msg['_tk'] = tk  # Cache'e al
            token_counts.append(tk)
            
        # Weighted knapsack problemi çözümü
        kept_indices = self._weighted_knapsack(weights, token_counts, max_tokens)
        
        # Dizine göre mesajları al
        kept = [history[i] for i in sorted(kept_indices)]
        
        # Cache temizliği
        for msg in kept:
            msg.pop('_tk', None)
            
        return kept

    def _identify_important_messages(self, history: List[Dict[str, str]]) -> Set[int]:
        """Önemli mesajları tanımla."""
        important_indices = set()
        
        for i, msg in enumerate(history):
            text = msg.get("text", "").lower()
            
            # Sistem mesajları önemlidir
            if msg.get("role") == "system":
                important_indices.add(i)
                continue
                
            # Çok önemli görünen mesajlar
            if any(marker in text for marker in ["önemli", "not", "hatırla", "unutma", "dikkat", "important", "note", "remember"]):
                important_indices.add(i)
                
            # URL içeren mesajlar
            if re.search(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text):
                important_indices.add(i)
                
            # Soru içeren mesajlar
            if '?' in text:
                important_indices.add(i)
                
            # İlk ve son mesajlar her zaman önemli
            if i == 0 or i == len(history) - 1:
                important_indices.add(i)
                
        return important_indices
        
    def _weighted_knapsack(self, weights: List[float], costs: List[int], budget: int) -> Set[int]:
        """
        Ağırlıklı knapsack problem çözümü:
        En değerli mesajları seçer ama bütçeyi aşmaz
        """
        n = len(weights)
        # Ağırlık/maliyet oranına göre sırala (verimlilik)
        items = [(i, weights[i]/costs[i]) for i in range(n) if costs[i] > 0]
        items.sort(key=lambda x: x[1], reverse=True)
        
        selected = set()
        total_cost = 0
        
        for idx, _ in items:
            if total_cost + costs[idx] <= budget:
                selected.add(idx)
                total_cost += costs[idx]
        
        return selected
    
    # -------------------------------------------------------------------
    # 3. İçerik Temelli Deduplication ve Optimizasyon
    # -------------------------------------------------------------------
    
    def deduplicate_history(self, 
                          history: List[Dict[str, str]], 
                          similarity_threshold: float = 85.0,
                          semantic_threshold: float = 80.0) -> List[Dict[str, str]]:
        """
        Gelişmiş deduplication sistemi:
        - Hem metin benzerliği hem semantik benzerlik için kontrol
        - Mesaj çiftleri arası gecikmeli benzerlik (tekrarlama)
        - Önemli ve son mesajları koruma mantığı
        """
        if len(history) <= 2:
            return history
            
        # Önemli mesajları belirle
        important_indices = self._identify_important_messages(history)
        
        deduped_history = []
        past_texts = []
        fingerprints = []  # Semantic fingerprints
        
        # Son beş mesajı her zaman tut
        preserve_recent = min(5, len(history))
        recent_indices = set(range(len(history) - preserve_recent, len(history)))
        
        for i, msg in enumerate(history):
            text = msg.get("text", "").strip()
            role = msg.get("role", "user")
            
            if not text:
                continue
                
            # Her zaman korunması gereken mesajları ekle
            if i in important_indices or i in recent_indices:
                deduped_history.append(msg)
                past_texts.append(text)
                continue
                
            # Yakın benzerlik kontrolü
            if self._is_similar_to_any(text, past_texts, similarity_threshold):
                continue
                
            deduped_history.append(msg)
            past_texts.append(text)
                
        return deduped_history
        
    def _is_similar_to_any(self, text: str, past_texts: List[str], threshold: float) -> bool:
        """Metnin geçmiş metinlere benzerliğini kontrol eder."""
        for past in past_texts:
            # Token sort ratio - kelime sırası farklı olsa bile benzerliği yakalar
            score = fuzz.token_sort_ratio(text, past)
            if score >= threshold:
                return True
                
            # Partial ratio - kısmi eşleşmeleri yakalar
            score = fuzz.partial_ratio(text, past)
            if score >= threshold + 5:  # Kısmi eşleşmeler için biraz daha yüksek eşik
                return True
                
        return False
        
    # -------------------------------------------------------------------
    # 4. Özellik Tabanlı Geçmiş Özeti
    # -------------------------------------------------------------------
    
    async def generate_enhanced_summary(self, 
                                      history: List[Dict[str, str]], 
                                      max_tokens: int = 700) -> Dict[str, Any]:
        """
        Gelişmiş özet oluşturma: 
        - Sohbet içinde bulunan ana başlıkları, varlıkları ve öğeleri çıkarır
        - Hem özet hem de anahtar öğeler tablosu oluşturur
        - Uzun dönem hafıza için bilgi yapısı hazırlar
        """
        if not history or len(history) < 8:  # Küçük sohbetler için özetleme yapmaz
            return {"summary": "", "entities": {}}
            
        # LLM için input hazırla
        filtered = [msg for msg in history if msg["role"] in ("user", "assistant")]
        recent = filtered[-12:] if len(filtered) > 12 else filtered
        
        # Sohbet içeriğini LLM için formatla
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['text'].strip()}" 
            for msg in recent if msg.get("text")
        )
        
        # Özetleme için gelişmiş prompt
        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "Kontekst analistliği görevini üstleneceksin. Aşağıdaki konuşmayı inceleyerek şu üç şeyi yapacaksın:\n"
                    "1. ÖZET: Konuşmanın ana konusunu ve önemli noktalarını özetleyen kısa bir paragraf (en fazla 200 kelime)\n"
                    "2. ANA KONULAR: Konuşmada geçen en önemli 3-5 anahtar konu/fikir listesi\n"
                    "3. VARLIKLAR: Konuşmada bahsedilen önemli varlıklar (kişiler, yerler, ürünler, kavramlar vb) ve bunların tanımı/açıklaması\n\n"
                    "Yanıtını JSON formatında döndür. Gereksiz açıklamalar yapma."
                )
            },
            {"role": "user", "content": conversation_text}
        ]
        
        try:
            # LLM'den özet ve analiz iste
            response = await self._generate_with_retry(summary_prompt)
            
            # JSON çıktısını parse etmeye çalış
            result = self._extract_json_from_text(response)
            if not result:
                # Fallback için basit özet döndürme
                return {"summary": response[:500], "entities": {}}
                
            # Önemli varlıkları entity memory'ye ekle
            if "VARLIKLAR" in result:
                for entity, description in result["VARLIKLAR"].items():
                    self.entity_memory[entity] = {
                        "description": description,
                        "last_mentioned": datetime.now(),
                        "mention_count": self.entity_memory.get(entity, {}).get("mention_count", 0) + 1
                    }
                    
            # Önemli kavramları ekle
            if "ANA_KONULAR" in result:
                for topic in result["ANA_KONULAR"]:
                    self.key_concepts.add(topic)
                    
            return {
                "summary": result.get("ÖZET", ""),
                "topics": result.get("ANA_KONULAR", []),
                "entities": result.get("VARLIKLAR", {})
            }
            
        except Exception as e:
            print(f"Özet oluşturma hatası: {str(e)}")
            # Basit fallback özet
            return {"summary": self._simple_summarize(history), "entities": {}}
            
    def _extract_json_from_text(self, text: str) -> Dict:
        """Metin içinden JSON yapıyı çıkarma."""
        try:
            # Önce tüm metni JSON olarak parse etmeyi dene
            return json.loads(text)
        except:
            # JSON blokunu regex ile bulmayı dene
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            json_match = re.search(json_pattern, text)
            
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass
                    
            # Son çare: Anahtar yapıların içeriğini çıkarmaya çalış
            result = {}
            
            # Özet bölümünü bul
            summary_pattern = r'(?:ÖZET|özet|Özet):\s*(.*?)(?:\n\n|\n[A-Z]|$)'
            summary_match = re.search(summary_pattern, text, re.DOTALL)
            if summary_match:
                result["ÖZET"] = summary_match.group(1).strip()
                
            return result
            
    def _simple_summarize(self, history: List[Dict[str, str]], max_words: int = 200) -> str:
        """LLM çağrısı başarısız olduğunda basit özet üretme."""
        texts = [msg.get("text", "") for msg in history[-10:] if msg.get("role") == "user"]
        combined = " ".join(texts)
        words = combined.split()
        if len(words) <= max_words:
            return combined
        return " ".join(words[:max_words]) + "..."
        
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def _generate_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """Retry mekanizması ile LLM çağrısı yapma."""
        if not self.client:
            return ""
            
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                options={
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "top_p": 0.9,
                }
            )
            
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"].strip()
            return ""
            
        except Exception as e:
            print(f"LLM çağrısı hatası: {str(e)}")
            raise
            
    # -------------------------------------------------------------------
    # 5. Ana EntryPoint - Gelişmiş History İşleme
    # -------------------------------------------------------------------
            
    async def process_history(self, 
                             history: List[Dict[str, str]], 
                             user_message: str,
                             force_summarize: bool = False) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Ana history işleme fonksiyonu:
        - Token hesaplaması ve trimming
        - Deduplication
        - Gerektiğinde özetleme
        - Bilgilerin durum bilgisi için döndürülmesi
        """
        # Metadata dizileri
        context_metadata = {}
        
        # Token hesaplama
        user_message_tokens = count_tokens(user_message)
        available_history_tokens = self.AVAILABLE_INPUT_TOKENS - user_message_tokens
        
        # İlk aşama trimming
        trimmed_history = self.trim_history_by_tokens(
            history, 
            max_tokens=available_history_tokens,
            decay_factor=0.92  # Eski mesajlar için decay faktörü
        )
        
        # Duplicate temizliği
        deduped_history = self.deduplicate_history(
            trimmed_history,
            similarity_threshold=85.0
        )
        
        # Özet gerekiyor mu kontrolü
        needs_summary = force_summarize or (len(deduped_history) > 15)
        summary_data = {"summary": "", "entities": {}, "topics": []}
        
        if needs_summary:
            summary_data = await self.generate_enhanced_summary(deduped_history)
            summary_text = summary_data.get("summary", "")
            
            if summary_text:
                # Özeti sistem mesajı olarak history başına ekle
                summary_msg = {
                    "role": "system",
                    "text": f"📌 Konuşma Özeti: {summary_text}"
                }
                
                # Eğer varlıklar varsa, onları da ekle
                entities = summary_data.get("entities", {})
                if entities:
                    entity_text = "\n📋 Önemli Varlıklar: "
                    for entity, desc in entities.items():
                        if isinstance(desc, str):
                            entity_text += f"\n- {entity}: {desc}"
                        else:
                            entity_text += f"\n- {entity}"
                    
                    summary_msg["text"] += entity_text
                
                # Özetleme sonrası durumu, kullanıcı mesajı önüne ekle
                deduped_history.insert(0, summary_msg)
                
                # Son bir kez daha token bütçesine göre kırpma
                deduped_history = self.trim_history_by_tokens(
                    deduped_history, 
                    max_tokens=available_history_tokens
                )
                
        # Metadata'yı hazırla
        context_metadata = {
            "tokens": {
                "user_message": user_message_tokens,
                "history": sum(count_tokens(msg.get("text", "")) for msg in deduped_history),
                "available": available_history_tokens,
                "total": self.AVAILABLE_INPUT_TOKENS
            },
            "stats": {
                "original_message_count": len(history),
                "final_message_count": len(deduped_history),
                "compression_ratio": len(deduped_history) / max(1, len(history)),
                "summary_applied": needs_summary
            },
            "summary": summary_data
        }
            
        return deduped_history, context_metadata
        
    # -------------------------------------------------------------------
    # 6. Uzun Dönem Hafıza Yönetimi (Long-Term Memory)
    # -------------------------------------------------------------------
    
    def update_entity_memory(self, message: Dict[str, str]) -> None:
        """Mesaj içeriğini analiz ederek varlık hafızasını günceller."""
        text = message.get("text", "").lower()
        
        # Mevcut varlıkları kontrol et
        for entity in list(self.entity_memory.keys()):  
            if entity.lower() in text:
                # Varlık mesajda geçiyorsa güncelle
                self.entity_memory[entity]["last_mentioned"] = datetime.now()
                self.entity_memory[entity]["mention_count"] += 1
    
    def get_relevant_entities(self, query: str, max_entities: int = 3) -> Dict[str, Any]:
        """Sorgu ile en alakalı varlıkları döndür."""
        if not self.entity_memory:
            return {}
            
        query = query.lower()
        scored_entities = []
        
        for entity, data in self.entity_memory.items():
            # Basit alaka skorlaması
            relevance = fuzz.partial_ratio(query, entity.lower()) / 100.0
            
            # Yenilik faktörü - yakın zamanda bahsedilen varlıklar daha önemli
            recency = 1.0
            if "last_mentioned" in data:
                time_diff = (datetime.now() - data["last_mentioned"]).total_seconds()
                # Son 5 dakika içinde bahsedildiyse maksimum yenilik
                recency = max(0.1, min(1.0, 300 / max(300, time_diff)))
                
            # Popülerlik faktörü - çok bahsedilen varlıklar daha önemli
            popularity = min(1.0, data.get("mention_count", 0) / 5)
            
            # Toplam skor
            total_score = (relevance * 0.6) + (recency * 0.3) + (popularity * 0.1)
            
            scored_entities.append((entity, data, total_score))
            
        # Skora göre sırala ve en üstteki varlıkları döndür
        scored_entities.sort(key=lambda x: x[2], reverse=True)
        
        result = {}
        for entity, data, score in scored_entities[:max_entities]:
            if score > 0.3:  # Minimum alaka eşiği
                result[entity] = data
                
        return result