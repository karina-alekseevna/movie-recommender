from typing import List, Optional
from dataclasses import dataclass
import json
import re
import hashlib
import pathlib
import numpy as np
from collections import OrderedDict

from backend.config import (
    SENTIMENT_MODEL_PATH,
    LLM_PROVIDER,
    LLM_MODEL_NAME,
    LLM_API_KEY,
    LLM_BASE_URL,
)
from backend.nlp.embedder import embedder

@dataclass
class SentimentResult:
    label: str   
    score: float   
    confidence: float
    method: str = ''

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'score': self.score,
            'confidence': self.confidence,
        }


_VALID_LABELS = frozenset(('positive', 'negative', 'neutral', 'mixed'))

SENTIMENT_SYSTEM_PROMPT = """\
Ты классификатор тональности отзывов на русском языке.

Определи ОТНОШЕНИЕ АВТОРА к произведению (не эмоции сюжета).
- Грустный сюжет + похвала = positive
- Красивые слова + критика = negative  
- Сарказм/ирония ("прекрасный способ убить время") = negative
- Похвала и критика поровну = mixed

Ответ — ТОЛЬКО JSON без markdown:
{"label":"positive|negative|neutral|mixed","score":<-1.0..1.0>,"confidence":<0.0..1.0>}"""

SENTIMENT_USER_TEMPLATE = "Тональность отзыва:\n\"{text}\""

BATCH_SYSTEM_PROMPT = """\
Ты классификатор тональности отзывов на русском языке.

Определи ОТНОШЕНИЕ АВТОРА к произведению (не эмоции сюжета).
- Грустный сюжет + похвала = positive
- Красивые слова + критика = negative
- Сарказм/ирония = negative
- Похвала и критика поровну = mixed

Ответ — ТОЛЬКО JSON-массив без markdown. Порядок элементов = порядок отзывов.
[{"label":"...","score":...,"confidence":...}, ...]"""

BATCH_USER_TEMPLATE = "Определи тональность каждого отзыва:\n{items}"

def _parse_llm_json(content: str) -> Optional[dict]:
    """Парсит единичный JSON-результат из ответа LLM."""
    content = re.sub(r'```(?:json)?\s*', '', content).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Попробуем найти JSON-объект в тексте
        match = re.search(r'\{[^{}]+\}', content)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    return _validate_sentiment_dict(data)


def _parse_llm_json_array(content: str, expected_count: int) -> Optional[List[dict]]:
    """Парсит массив JSON-результатов из ответа LLM."""
    content = re.sub(r'```(?:json)?\s*', '', content).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'$.*$', content, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    if not isinstance(data, list) or len(data) != expected_count:
        return None

    results = []
    for item in data:
        validated = _validate_sentiment_dict(item)
        if validated is None:
            return None
        results.append(validated)
    return results


def _validate_sentiment_dict(data: dict) -> Optional[dict]:
    """Валидирует и нормализует словарь с результатом."""
    if not isinstance(data, dict):
        return None

    label = str(data.get('label', 'neutral')).lower().strip()
    if label not in _VALID_LABELS:
        label = 'neutral'

    try:
        score = float(data.get('score', 0.0))
    except (ValueError, TypeError):
        score = 0.0
    score = max(-1.0, min(1.0, score))

    try:
        confidence = float(data.get('confidence', 0.5))
    except (ValueError, TypeError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # Согласованность label и score
    if label == 'positive' and score < 0:
        score = abs(score)
    elif label == 'negative' and score > 0:
        score = -abs(score)

    return {
        'label': label,
        'score': round(score, 3),
        'confidence': round(confidence, 3),
    }

class LLMClient:
    """Базовый класс для LLM-клиентов."""

    def query(self, text: str) -> Optional[dict]:
        raise NotImplementedError

    def query_batch(self, texts: List[str]) -> Optional[List[dict]]:
        """Батч-запрос. По умолчанию — последовательные одиночные."""
        return None  # Подклассы могут переопределить


class OllamaClient(LLMClient):

    def __init__(self, model: str, base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self._available: Optional[bool] = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import httpx
            resp = httpx.get(f'{self.base_url}/api/tags', timeout=3.0)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def _request(self, system: str, user: str, max_tokens: int = 200) -> Optional[str]:
        if not self._check_available():
            return None
        try:
            import httpx
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
                'stream': False,
                'options': {
                    'temperature': 0.05,
                    'num_predict': max_tokens,
                }
            }
            resp = httpx.post(
                f'{self.base_url}/api/chat',
                json=payload,
                timeout=20.0,
            )
            resp.raise_for_status()
            return resp.json()['message']['content']
        except Exception as e:
            print(f"[Sentiment] Ollama error: {e}")
            self._available = None  # Сбрасываем — проверим заново
            return None

    def query(self, text: str) -> Optional[dict]:
        content = self._request(
            SENTIMENT_SYSTEM_PROMPT,
            SENTIMENT_USER_TEMPLATE.format(text=text[:1000]),
        )
        if content is None:
            return None
        return _parse_llm_json(content)

    def query_batch(self, texts: List[str]) -> Optional[List[dict]]:
        if len(texts) <= 1:
            return None

        # Ограничиваем батч до 10 — иначе LLM путается
        if len(texts) > 10:
            return None

        items = "\n".join(
            f'{i + 1}. "{t[:300]}"' for i, t in enumerate(texts)
        )
        content = self._request(
            BATCH_SYSTEM_PROMPT,
            BATCH_USER_TEMPLATE.format(items=items),
            max_tokens=100 * len(texts),
        )
        if content is None:
            return None
        return _parse_llm_json_array(content, len(texts))


class OpenAIClient(LLMClient):

    def __init__(self, model: str, api_key: str, base_url: str = 'https://api.openai.com/v1'):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    def _request(self, system: str, user: str, max_tokens: int = 200) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            import httpx
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
                'temperature': 0.05,
                'max_tokens': max_tokens,
            }
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            resp = httpx.post(
                f'{self.base_url}/chat/completions',
                json=payload,
                headers=headers,
                timeout=20.0,
            )
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"[Sentiment] OpenAI error: {e}")
            return None

    def query(self, text: str) -> Optional[dict]:
        content = self._request(
            SENTIMENT_SYSTEM_PROMPT,
            SENTIMENT_USER_TEMPLATE.format(text=text[:1000]),
        )
        if content is None:
            return None
        return _parse_llm_json(content)

    def query_batch(self, texts: List[str]) -> Optional[List[dict]]:
        if len(texts) <= 1 or len(texts) > 15:
            return None

        items = "\n".join(
            f'{i + 1}. "{t[:300]}"' for i, t in enumerate(texts)
        )
        content = self._request(
            BATCH_SYSTEM_PROMPT,
            BATCH_USER_TEMPLATE.format(items=items),
            max_tokens=80 * len(texts),
        )
        if content is None:
            return None
        return _parse_llm_json_array(content, len(texts))


class GigaChatClient(LLMClient):

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._token: Optional[str] = None
        self._token_expires: float = 0

    def _get_token(self) -> str:
        import httpx
        import time
        import uuid

        if self._token and time.time() < self._token_expires:
            return self._token

        resp = httpx.post(
            'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
            headers={
                'Authorization': f'Basic {self.api_key}',
                'RqUID': str(uuid.uuid4()),
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            data={'scope': 'GIGACHAT_API_PERS'},
            verify=False,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data['access_token']
        self._token_expires = data['expires_at'] / 1000 - 60
        return self._token

    def _request(self, system: str, user: str, max_tokens: int = 200) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            import httpx
            token = self._get_token()
            payload = {
                'model': 'GigaChat',
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
                'temperature': 0.05,
                'max_tokens': max_tokens,
            }
            resp = httpx.post(
                'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
                json=payload,
                headers={'Authorization': f'Bearer {token}'},
                verify=False,
                timeout=20.0,
            )
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"[Sentiment] GigaChat error: {e}")
            return None

    def query(self, text: str) -> Optional[dict]:
        content = self._request(
            SENTIMENT_SYSTEM_PROMPT,
            SENTIMENT_USER_TEMPLATE.format(text=text[:1000]),
        )
        if content is None:
            return None
        return _parse_llm_json(content)

class SemanticFallback:
    """
    Семантический анализ через эмбеддинги — используется
    когда LLM недоступна.
    """

    def __init__(self):
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        pos_refs = [
            "отличный фильм, очень понравился, рекомендую всем",
            "шедевр, потрясающая игра актёров, великолепно снято",
            "трогательный и глубокий фильм, заставляет задуматься",
            "один из лучших фильмов что я видел, обязательно к просмотру",
            "прекрасная режиссура, восхитительный сюжет, браво",
            "невозможно оторваться, смотрится на одном дыхании",
            "фильм берёт за душу, хочется пересматривать снова",
            "великолепная музыка и атмосфера, получил удовольствие",
            "очень сильное кино, до мурашек, лучшее за год",
            "тонкий юмор, умный сценарий, актёры на высоте",
            "книга потрясающая, читается легко и увлекательно",
            "прекрасный альбом, каждый трек хит",
        ]

        neg_refs = [
            "ужасный фильм, полное разочарование, не рекомендую",
            "скучно, бессмысленно, пустая трата времени",
            "отвратительная игра актёров, слабый сценарий",
            "раздражает, невыносимо смотреть эту халтуру",
            "провал, дешёвка, деньги на ветер",
            "предсказуемый и банальный, клише на клише",
            "не смог досмотреть, выключил на середине",
            "худший фильм что я видел, полный отстой",
            "затянуто, нудно, сюжет никакой",
            "разочарован, ожидал большего, переоценённый фильм",
            "книга скучная, еле дочитал через силу",
            "музыка однообразная, ни одного запоминающегося трека",
        ]

        neutral_refs = [
            "обычный фильм, ничего особенного, средне",
            "посмотрел и забыл, не плохо не хорошо",
            "нормальное кино для вечера, без претензий",
            "стандартный представитель жанра, ровно",
            "есть плюсы и минусы, в целом нормально",
        ]

        self.pos_vecs = embedder.encode(pos_refs, normalize=True)
        self.neg_vecs = embedder.encode(neg_refs, normalize=True)
        self.neutral_vecs = embedder.encode(neutral_refs, normalize=True)

        self.pos_centroid = np.mean(self.pos_vecs, axis=0)
        self.neg_centroid = np.mean(self.neg_vecs, axis=0)
        self.neutral_centroid = np.mean(self.neutral_vecs, axis=0)

        self._initialized = True

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(
                label='neutral', score=0.0, confidence=0.0,
                method='fallback_empty'
            )

        self._ensure_initialized()

        vec = embedder.encode(text, normalize=True)
        if vec.ndim > 1:
            vec = vec[0]

        # Средняя близость к top-3 ближайшим эталонам каждого класса
        pos_sim = self._top_k_similarity(vec, self.pos_vecs, k=3)
        neg_sim = self._top_k_similarity(vec, self.neg_vecs, k=3)
        neu_sim = self._top_k_similarity(vec, self.neutral_vecs, k=2)

        # Также считаем расстояние до центроидов
        pos_cent = float(np.dot(vec, self.pos_centroid))
        neg_cent = float(np.dot(vec, self.neg_centroid))
        neu_cent = float(np.dot(vec, self.neutral_centroid))

        # Комбинированный скор: 60% top-k, 40% centroid
        pos_final = 0.6 * pos_sim + 0.4 * pos_cent
        neg_final = 0.6 * neg_sim + 0.4 * neg_cent
        neu_final = 0.6 * neu_sim + 0.4 * neu_cent

        diff = pos_final - neg_final

        # Определяем label
        if neu_final > pos_final and neu_final > neg_final and abs(diff) < 0.05:
            label = 'neutral'
            score = round(diff * 2, 3)
            confidence = round(min(0.6, neu_final), 3)
        elif diff > 0.02:
            label = 'positive'
            score = round(min(0.95, diff * 4), 3)
            confidence = round(min(0.7, diff * 3 + 0.15), 3)
        elif diff < -0.02:
            label = 'negative'
            score = round(max(-0.95, diff * 4), 3)
            confidence = round(min(0.7, abs(diff) * 3 + 0.15), 3)
        else:
            label = 'neutral'
            score = round(diff * 2, 3)
            confidence = round(0.2 + abs(diff), 3)

        return SentimentResult(
            label=label,
            score=score,
            confidence=confidence,
            method='semantic_fallback',
        )

    @staticmethod
    def _top_k_similarity(vec: np.ndarray, ref_vecs: np.ndarray, k: int = 3) -> float:
        """Средняя косинусная близость к top-k ближайшим эталонам."""
        sims = ref_vecs @ vec
        top_k = np.sort(sims)[-k:]
        return float(np.mean(top_k))

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Батч-анализ через эмбеддинги — эффективнее чем по одному."""
        if not texts:
            return []

        self._ensure_initialized()

        valid_indices = []
        valid_texts = []
        results: List[Optional[SentimentResult]] = [None] * len(texts)

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = SentimentResult(
                    label='neutral', score=0.0, confidence=0.0,
                    method='fallback_empty'
                )
            else:
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            return results

        # Кодируем все тексты одним батчем
        vecs = embedder.encode(valid_texts, normalize=True)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        # Матричное умножение — все сходства за раз
        pos_sims = vecs @ self.pos_vecs.T  # (N, num_pos)
        neg_sims = vecs @ self.neg_vecs.T  # (N, num_neg)
        neu_sims = vecs @ self.neutral_vecs.T  # (N, num_neu)

        pos_centroids = vecs @ self.pos_centroid  # (N,)
        neg_centroids = vecs @ self.neg_centroid
        neu_centroids = vecs @ self.neutral_centroid

        for idx, orig_idx in enumerate(valid_indices):
            pos_top = float(np.mean(np.sort(pos_sims[idx])[-3:]))
            neg_top = float(np.mean(np.sort(neg_sims[idx])[-3:]))
            neu_top = float(np.mean(np.sort(neu_sims[idx])[-2:]))

            pos_final = 0.6 * pos_top + 0.4 * float(pos_centroids[idx])
            neg_final = 0.6 * neg_top + 0.4 * float(neg_centroids[idx])
            neu_final = 0.6 * neu_top + 0.4 * float(neu_centroids[idx])

            diff = pos_final - neg_final

            if neu_final > pos_final and neu_final > neg_final and abs(diff) < 0.05:
                label = 'neutral'
                score = round(diff * 2, 3)
                confidence = round(min(0.6, neu_final), 3)
            elif diff > 0.02:
                label = 'positive'
                score = round(min(0.95, diff * 4), 3)
                confidence = round(min(0.7, diff * 3 + 0.15), 3)
            elif diff < -0.02:
                label = 'negative'
                score = round(max(-0.95, diff * 4), 3)
                confidence = round(min(0.7, abs(diff) * 3 + 0.15), 3)
            else:
                label = 'neutral'
                score = round(diff * 2, 3)
                confidence = round(0.2 + abs(diff), 3)

            results[orig_idx] = SentimentResult(
                label=label, score=score, confidence=confidence,
                method='semantic_fallback',
            )

        return results

class SentimentCache:
    """LRU-кэш с нормализованными ключами."""

    def __init__(self, max_size: int = 3000):
        self._cache: OrderedDict[str, SentimentResult] = OrderedDict()
        self._max_size = max_size

    @staticmethod
    def _key(text: str) -> str:
        """Хэш нормализованного текста — экономит память."""
        normalized = ' '.join(text.lower().split())[:500]
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[SentimentResult]:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, text: str, result: SentimentResult):
        key = self._key(text)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def get_many(self, texts: List[str]) -> List[Optional[SentimentResult]]:
        return [self.get(t) for t in texts]

    def put_many(self, texts: List[str], results: List[SentimentResult]):
        for t, r in zip(texts, results):
            self.put(t, r)

class SentimentAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.llm_client = self._create_llm_client()
        self.fallback = SemanticFallback()
        self.cache = SentimentCache(max_size=3000)

        self._initialized = True

        llm_name = type(self.llm_client).__name__ if self.llm_client else 'None'
        print(f"[Sentiment] Initialized v6 — LLM: {llm_name}")

    def _create_llm_client(self) -> Optional[LLMClient]:
        provider = str(getattr(LLM_PROVIDER, '__str__', lambda: 'none')()
                       if callable(getattr(LLM_PROVIDER, '__str__', None))
                       else LLM_PROVIDER).lower()

        if provider == 'ollama':
            base_url = str(LLM_BASE_URL) if LLM_BASE_URL else 'http://localhost:11434'
            model = str(LLM_MODEL_NAME) if LLM_MODEL_NAME else 'gemma2:9b'
            client = OllamaClient(model=model, base_url=base_url)
            print(f"[Sentiment] Using Ollama: {model} @ {base_url}")
            return client

        elif provider == 'openai':
            api_key = str(LLM_API_KEY) if LLM_API_KEY else ''
            model = str(LLM_MODEL_NAME) if LLM_MODEL_NAME else 'gpt-4o-mini'
            base_url = str(LLM_BASE_URL) if LLM_BASE_URL else 'https://api.openai.com/v1'
            if api_key:
                print(f"[Sentiment] Using OpenAI-compatible: {model}")
                return OpenAIClient(model=model, api_key=api_key, base_url=base_url)

        elif provider == 'gigachat':
            api_key = str(LLM_API_KEY) if LLM_API_KEY else ''
            if api_key:
                print("[Sentiment] Using GigaChat")
                return GigaChatClient(api_key=api_key)

        print("[Sentiment] No LLM — semantic fallback only")
        return None

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(
                label='neutral', score=0.0, confidence=0.0, method='empty'
            )

        cached = self.cache.get(text)
        if cached is not None:
            return cached

        result = self._analyze_with_llm(text)
        if result is None:
            result = self.fallback.analyze(text)

        self.cache.put(text, result)
        return result

    def _analyze_with_llm(self, text: str) -> Optional[SentimentResult]:
        if self.llm_client is None:
            return None

        response = self.llm_client.query(text)
        if response is None:
            return None

        return SentimentResult(
            label=response['label'],
            score=response['score'],
            confidence=response['confidence'],
            method=f'llm_{type(self.llm_client).__name__}',
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Пакетный анализ:
        1. Проверяем кэш для всех текстов
        2. Некэшированные отправляем в LLM батчем (если поддерживается)
        3. Оставшиеся — через фоллбэк батчем
        """
        if not texts:
            return []

        results: List[Optional[SentimentResult]] = [None] * len(texts)
        uncached_indices: List[int] = []

        # 1. Кэш
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = SentimentResult(
                    label='neutral', score=0.0, confidence=0.0, method='empty'
                )
            else:
                cached = self.cache.get(text)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)

        if not uncached_indices:
            return results

        # 2. LLM батч
        llm_failed_indices: List[int] = []
        uncached_texts = [texts[i] for i in uncached_indices]

        if self.llm_client is not None:
            batch_result = self.llm_client.query_batch(uncached_texts)

            if batch_result is not None and len(batch_result) == len(uncached_indices):
                # Батч успешен
                for idx, data in zip(uncached_indices, batch_result):
                    r = SentimentResult(
                        label=data['label'],
                        score=data['score'],
                        confidence=data['confidence'],
                        method=f'llm_batch_{type(self.llm_client).__name__}',
                    )
                    results[idx] = r
                    self.cache.put(texts[idx], r)
            else:
                for i, orig_idx in enumerate(uncached_indices):
                    response = self.llm_client.query(uncached_texts[i])
                    if response is not None:
                        r = SentimentResult(
                            label=response['label'],
                            score=response['score'],
                            confidence=response['confidence'],
                            method=f'llm_{type(self.llm_client).__name__}',
                        )
                        results[orig_idx] = r
                        self.cache.put(texts[orig_idx], r)
                    else:
                        llm_failed_indices.append(orig_idx)
        else:
            llm_failed_indices = uncached_indices

        if llm_failed_indices:
            fallback_texts = [texts[i] for i in llm_failed_indices]
            fallback_results = self.fallback.analyze_batch(fallback_texts)

            for idx, r in zip(llm_failed_indices, fallback_results):
                results[idx] = r
                self.cache.put(texts[idx], r)

        return results


sentiment_analyzer = SentimentAnalyzer()
