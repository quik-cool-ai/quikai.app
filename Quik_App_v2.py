from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tiktoken
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime
import ollama
import aiohttp
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
from rapidfuzz import fuzz
from typing import List, Tuple, Dict, Any, Optional
from random import choice, uniform
from time import time
from werkzeug.utils import secure_filename
import os
from asgiref.wsgi import WsgiToAsgi
from history_manager import HistoryManager
from functools import lru_cache

# -------------------------------------------------------------------
# 1. Temel Ayarlar ve Global Sabitler
# -------------------------------------------------------------------

# GPU kullanımını zorla
os.environ["OLLAMA_FORCE_CUDA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Flask app // CORS // ASGI
app = Flask(__name__)
CORS(app)
asgi_app = WsgiToAsgi(app)

# Ollama client - Ollama model adı - Model sıcaklık
client = ollama.Client()
MODEL_NAME = "Quik_v2:latest"
DEFAULT_TEMPERATURE = 1.0

# HistoryManager initialization
history_manager = HistoryManager(
    total_max_tokens=131072,
    reserved_output_tokens=65536,
    client=client,
    model_name=MODEL_NAME
)

# User Agent listesi
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (Android 13; Mobile; rv:68.0) Gecko/68.0 Firefox/68.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
]

# Token sınırları (128k context için)
TOTAL_MAX_TOKENS: int = 131072
RESERVED_OUTPUT_TOKENS: int = 65536
AVAILABLE_INPUT_TOKENS: int = TOTAL_MAX_TOKENS - RESERVED_OUTPUT_TOKENS

# tiktoken için
tokenizer = tiktoken.get_encoding("cl100k_base")

# -------------------------------------------------------------------
# 2. Yardımcı Fonksiyonlar
# -------------------------------------------------------------------

@lru_cache(maxsize=16_384)
def count_tokens(text: str) -> int:
    """LRU-cache lets us avoid re-encoding identical strings across requests."""
    return len(tokenizer.encode(text))

def summarize_text(text: str, max_words: int = 1000) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

def append_current_date(query: str) -> str:
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    return f"{query} {current_date}"

def normalize_text_preserving_numbers(text: str) -> str:
    tokens = re.split(r'(\d+(?:\.\d+)?)', text)
    normalized_tokens = [
        token if re.fullmatch(r'\d+(?:\.\d+)?', token) else token.lower().strip()
        for token in tokens
    ]
    return " ".join(filter(None, normalized_tokens))

# -------------------------------------------------------------------
# 3. Asenkron Web İşlemleri ve Arama Sınıfı
# -------------------------------------------------------------------

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
async def async_fetch_url_content(url: str) -> str:
    try:
        headers = {
            "User-Agent": choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "DNT": "1"
        }
        async with aiohttp.ClientSession(headers=headers, trust_env=True) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response.raise_for_status()
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                for elem in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
                    elem.decompose()
                text = soup.get_text(separator=" ").strip()
                text = re.sub(r"\s+", " ", text)
                text = re.sub(r"(\n\s*){2,}", "\n", text)
                return text[:10000] if len(text) > 10000 else text
    except Exception as e:
        return ""

class SimpleCache:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.store = {}

    def get(self, key: str):
        entry = self.store.get(key)
        if entry:
            timestamp, value = entry
            if time() - timestamp < self.ttl:
                return value
            else:
                del self.store[key]
        return None

    def set(self, key: str, value):
        self.store[key] = (time(), value)

class SuperAdvancedWebSearchFree:
    def __init__(self) -> None:
        self.duckduckgo_url = "https://html.duckduckgo.com/html/"
        self.bing_url = "https://www.bing.com/search"
        self.headers = {
            "User-Agent": choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate"
        }
        self.session = aiohttp.ClientSession(headers=self.headers, trust_env=True)
        self.last_request_time = 0
        self.request_count = 0
        self.max_requests_per_minute = 10
        self.cache = SimpleCache(ttl=300)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def _rate_limit_check(self) -> None:
        current_time = time()
        if current_time - self.last_request_time >= 60:
            self.request_count = 0
            self.last_request_time = current_time
        if self.request_count >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.last_request_time)
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.last_request_time = time()
        self.request_count += 1

    async def _fetch_html(self, url: str, params: dict, method: str = "GET") -> Optional[str]:
        await self._rate_limit_check()
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            self.headers["User-Agent"] = choice(USER_AGENTS)
            if method.upper() == "POST":
                async with self.session.post(url, data=params, headers=self.headers, timeout=timeout) as response:
                    response.raise_for_status()
                    return await response.text()
            async with self.session.get(url, params=params, headers=self.headers, timeout=timeout) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            await asyncio.sleep(2)
            return None

    async def async_search_duckduckgo(self, query: str, num_results: int = 5) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        html = await self._fetch_html(self.duckduckgo_url, {"q": query}, method="POST")
        if not html:
            return results
        soup = BeautifulSoup(html, "html.parser")
        for result_div in soup.find_all("div", class_="result"):
            link = result_div.find("a", class_="result__a")
            if not link:
                continue
            title = link.get_text().strip()
            snippet_elem = result_div.find("a", class_="result__snippet")
            if not snippet_elem:
                snippet_elem = result_div.find("div", class_="result__snippet")
            snippet = snippet_elem.get_text().strip() if snippet_elem else ""
            url = link.get("href", "").strip()
            if not url.startswith("http"):
                continue
            results.append((title, snippet[:200], url))
            if len(results) >= num_results:
                break
        return results

    async def async_search_bing(self, query: str, num_results: int = 5) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        html = await self._fetch_html(self.bing_url, {"q": query})
        if not html:
            return results
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all("li", {"class": "b_algo"}):
            header = item.find("h2")
            if not header:
                continue
            link = header.find("a")
            if not link:
                continue
            title = link.get_text().strip()
            url = link.get("href", "").strip()
            snippet_elem = item.find("p")
            snippet = snippet_elem.get_text().strip() if snippet_elem else ""
            if not url.startswith("http"):
                continue
            results.append((title, snippet[:200], url))
            if len(results) >= num_results:
                break
        return results

    async def fetch_and_extract(self, url: str) -> str:
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with self.session.get(url, headers=self.headers, timeout=timeout) as response:
                response.raise_for_status()
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    tag.decompose()
                main_candidates = [
                    soup.find("article"),
                    soup.find("main"),
                    soup.find("div", id="content")
                ]
                content = ""
                for candidate in main_candidates:
                    if candidate:
                        content = candidate.get_text(separator=" ", strip=True)
                        if len(content) > 100:
                            break
                if not content:
                    content = soup.get_text(separator=" ", strip=True)
                content = re.sub(r"\s+", " ", content).strip()
                return content[:10000] if len(content) > 10000 else content
        except Exception as e:
            return ""

    def rank_results(self, query: str, results: List[Tuple[str, str, str, Optional[str]]]) -> List[Tuple[str, str, str, Optional[str]]]:
        ranked = []
        query_norm = normalize_text_preserving_numbers(query)
        for title, snippet, url, content in results:
            norm_title = normalize_text_preserving_numbers(title)
            norm_snippet = normalize_text_preserving_numbers(snippet)
            title_score = fuzz.partial_ratio(query_norm, norm_title) * 0.5
            snippet_score = fuzz.partial_ratio(query_norm, norm_snippet) * 0.3
            content_score = fuzz.partial_ratio(query_norm, normalize_text_preserving_numbers(content)) * 0.2 if content else 0
            total_score = title_score + snippet_score + content_score
            ranked.append((total_score, title, snippet, url, content))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [(title, snippet, url, content) for _, title, snippet, url, content in ranked]

    def filter_duplicates(self, results: List[Tuple[str, str, str, Optional[str]]]) -> List[Tuple[str, str, str, Optional[str]]]:
        seen = set()
        unique_results = []
        for title, snippet, url, content in results:
            normalized_title = normalize_text_preserving_numbers(title)
            if normalized_title not in seen and len(snippet) > 20:
                unique_results.append((title, snippet, url, content))
                seen.add(normalized_title)
        return unique_results

    async def async_super_advanced_search(self, query: str, num_results: int = 10) -> List[Tuple[str, str, str, Optional[str]]]:
        query_with_date = append_current_date(query)
        cached_result = self.cache.get(query_with_date)
        if cached_result is not None:
            return cached_result

        await asyncio.sleep(uniform(0.1, 0.3))
        self.headers["User-Agent"] = choice(USER_AGENTS)
        ddg_task = self.async_search_duckduckgo(query_with_date, num_results=(num_results // 2 + 1))
        bing_task = self.async_search_bing(query_with_date, num_results=(num_results // 2 + 1))
        ddg_results, bing_results = await asyncio.gather(ddg_task, bing_task)
        all_results = ddg_results + bing_results
        if not all_results:
            return [("Arama başarısız", "Web araması sonuç döndürmedi.", "", None)]

        async def fetch_result_content(result: Tuple[str, str, str]) -> Tuple[str, str, str, Optional[str]]:
            title, snippet, url = result
            content = await self.fetch_and_extract(url) if url else ""
            return (title, snippet, url, content)
        
        tasks = [fetch_result_content(result) for result in all_results]
        results_with_content = await asyncio.gather(*tasks)
        ranked_results = self.rank_results(query_with_date, results_with_content)
        unique_results = self.filter_duplicates(ranked_results)
        final_results = unique_results[:num_results]
        self.cache.set(query_with_date, final_results)
        return final_results

# -------------------------------------------------------------------
# 4. Flask Chat Endpoint
# -------------------------------------------------------------------

def extract_url(text: str) -> Optional[str]:
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*')
    match = url_pattern.search(text)
    return match.group(0) if match else None

@app.route("/chat", methods=["POST"])
async def chat() -> Response:
    try:
        print("=== /chat endpoint çağrıldı ===")
        req_data: Dict[str, Any] = request.get_json()
        print("Gelen istek:", req_data)
        user_message: str = req_data.get("message", "")
        history: List[Dict[str, str]] = req_data.get("history", [])
        enable_web_search: bool = req_data.get("enable_web_search", False)
        temperature: float = req_data.get("temperature", DEFAULT_TEMPERATURE)

        if not user_message:
            print("Hata: Mesaj sağlanmadı")
            return jsonify({"reply": "No message provided"}), 400

        # Tarih formatını standart hale getir
        history = [
            {"role": msg["sender"], "text": msg["text"]}
            for msg in history
            if msg.get("sender") in ["user", "assistant"] and msg.get("text")
        ]

        print("URL kontrol ediliyor")
        url = extract_url(user_message)
        website_content = ""
        if url:
            print(f"URL'den içerik çekiliyor: {url}")
            website_content = await async_fetch_url_content(url)
            if not website_content:
                website_content = "Unable to fetch content from the provided URL."
            instruction = user_message.replace(url, "").strip()
            user_message = (
                f"{instruction} based on this content from {url}:\n\n{website_content}"
                if instruction else f"Here is the content from {url}:\n\n{website_content}"
            )

        print("Kullanıcı mesajı özetleniyor")
        max_words = (AVAILABLE_INPUT_TOKENS // 4) * 0.75
        user_message = summarize_text(user_message, max_words=int(max_words))

        print("HistoryManager ile geçmiş işleniyor")
        processed_history, context_metadata = await history_manager.process_history(
            history,
            user_message,
            force_summarize=(len(history) > 15)
        )
        print("Geçmiş metadata:", context_metadata)

        print("Web arama kontrolü")
        search_results_text = ""
        if enable_web_search and not url:
            try:
                async with SuperAdvancedWebSearchFree() as websearch:
                    print("Web araması yapılıyor")
                    results = await websearch.async_super_advanced_search(user_message, num_results=5)
                    if results:
                        search_results_text = "Web Arama Sonuçları:\n"
                        for title, snippet, url, _ in results:
                            search_results_text += f"{title}\nÖzet: {snippet}\nURL: {url}\n\n"
                    else:
                        search_results_text = "Web Arama Sonuçları: Arama sonuç döndürmedi.\n"
            except Exception as e:
                print(f"Web arama hatası: {str(e)}")
                search_results_text = "Web Arama Sonuçları: Hata oluştu.\n"

        if search_results_text:
            user_message += (
                "\n\nNot: Kullanıcının talebiyle ilgili yapılan güncel web araması sonuçları aşağıda listelenmiştir. "
                "Cevabını bu bilgilere dayanarak oluşturabilirsin.\n" + search_results_text.strip()
            )

        STATIC_SYSTEM_PROMPT = """  
        Your name is Quik. You are an AI assistant developed by SGDD-ASAM.  
        You are not human, but you speak and act like one — naturally, intelligently, and kindly.  
        You assist users by writing code, explaining technical topics, summarizing web content, and analyzing documents.  
        You ALWAYS respond in the user's language, speak clearly, and avoid unnecessary repetition.  
        
        Do not mention that you were developed by SGDD-ASAM unless directly asked.  
        Do not explain your own identity unless the user explicitly requests it.  
        Never refer to yourself in the third person.  
        Avoid repeating greetings like 'Hello' or 'Hi' unless the user starts with one. Keep replies direct and to the point.
        Stay in character: You are Quik. You are helpful, trustworthy, and professional.
        """

        # Mesajları Quik formatına getir
        def format_history(history):
            formatted = [{"role": "system", "content": STATIC_SYSTEM_PROMPT.strip()}]
            formatted += [{"role": h["role"], "content": h["text"]} for h in history]
            return formatted

        print("Mesajlar formatlanıyor")
        messages = format_history(processed_history)
        messages.append({"role": "user", "content": user_message})

        print("Entity memory güncelleniyor")
        history_manager.update_entity_memory({"text": user_message})

        print("Streaming yanıtı başlatılıyor")
        def generate() -> str:
            try:
                stream = client.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "top_k": 64,
                        "top_p": 0.95,
                        "min_p": 0.0,
                        "repeat_last_n": 512,
                        "repeat_penalty": 1.4,
                        "presence_penalty": 0.9,
                        "frequency_penalty": 0.8,
                        "stop": ["<end_of_turn>"],
                        "max_tokens": RESERVED_OUTPUT_TOKENS
                    }
                )

                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield f"data: {json.dumps({'text': chunk['message']['content']})}\n\n"
                    elif "error" in chunk:
                        yield f"data: error: {json.dumps(chunk['error'])}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                print(f"Streaming hatası: {str(e)}")
                yield f"data: error: {str(e)}\n\n"

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["X-Accel-Buffering"] = "no"
        print("Yanıt gönderildi")
        return response

    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}")
        return jsonify({"reply": f"Beklenmeyen bir hata oluştu: {str(e)}"}), 500

# -------------------------------------------------------------------
# 4.1 Başlık Üretme
# -------------------------------------------------------------------

@app.route("/generate-title", methods=["POST"])
async def generate_title() -> Response:
    try:
        req_data: Dict[str, Any] = request.get_json()
        user_message: str = req_data.get("message", "")
        if not user_message:
            return jsonify({"title": "Yeni Sohbet"}), 200

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at generating short, meaningful, and creative conversation titles.\n"
                    "Read the user's message and generate a 3-word title that directly reflects the topic of the message.\n"
                    "Do not include greetings, questions, or generic phrases. Just return the title itself.\n"
                    "The title will be used as a label for this chat. Create the title in the language the user wrote in."
                )
            },
            {
                "role": "user",
                "content": user_message.strip()
            }
        ]

        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
            options={
                "stop": ["<end_of_turn>"],
                "temperature": 0.7,
                "top_k": 64,
                "top_p": 0.95,
                "min_p": 0.0,
                "repeat_last_n": 256,
                "repeat_penalty": 1.2,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
                "max_tokens": 15
            }
        )

        title = response["message"]["content"].strip()
        title = re.sub(r"(system:)", "", title, flags=re.IGNORECASE).strip()

        words = title.split()
        if len(words) > 3:
            title = ' '.join(words[:3])

        title = re.sub(r"[^\w\s-]", "", title)

        return jsonify({"title": title}), 200

    except Exception as e:
        return jsonify({"title": f"Hata: {str(e)}"}), 500

# -------------------------------------------------------------------
# 5. Dosya yükleme
# -------------------------------------------------------------------

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı.'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'Dosya listesi boş.'}), 400

    combined_text = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()

            print("Processing file:", filename)

            file.seek(0)

            if ext == 'pdf':
                try:
                    import pdfplumber
                    file.seek(0)
                    with pdfplumber.open(file) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        combined_text.append(text)
                except Exception as e:
                    combined_text.append(f"PDF okunurken hata oluştu: {str(e)}")
            elif ext in ['doc', 'docx']:
                try:
                    import docx
                    file.seek(0)
                    doc = docx.Document(file)
                    text = "\n".join(para.text for para in doc.paragraphs if para.text)
                    combined_text.append(text)
                except Exception as e:
                    combined_text.append(f"DOC/DOCX okunurken hata oluştu: {str(e)}")
            elif ext == 'txt':
                try:
                    file.seek(0)
                    text = file.read().decode('utf-8')
                    combined_text.append(text)
                except Exception as e:
                    combined_text.append(f"TXT okunurken hata oluştu: {str(e)}")

    full_text = "\n".join(filter(None, combined_text))
    return jsonify({'content': full_text})

# -------------------------------------------------------------------
# 5. Uygulama Başlatma
# -------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)