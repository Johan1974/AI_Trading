"""
BESTANDSNAAM: app/ai/sentiment/finbert_sentiment.py
FUNCTIE: Sentiment analyzer met HuggingFace Transformers FinBERT.
         Bevat ook RedisSentimentReader voor de news-container architectuur.
"""

import json
import os
import threading
from collections.abc import Sequence

try:
    import torch
    from transformers import pipeline
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    pipeline = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from app.ai.base import SentimentAnalyzer
from app.ai.types import SentimentResult


class FinBertSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, model_id: str = "ProsusAI/finbert") -> None:
        self.model_id = model_id
        self.model_name = f"hf-{model_id}"
        use_cuda = str(os.getenv("FINBERT_USE_CUDA", "0")).strip().lower() in ("1", "true", "yes")
        if use_cuda:
            try:
                device_id = 0 if torch.cuda.is_available() else -1
            except Exception:
                device_id = -1
        else:
            device_id = -1
        self._pipe = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
            truncation=True,
            max_length=256,
            device=device_id,
        )

    def score(self, texts: Sequence[str]) -> SentimentResult:
        scored = self.score_with_breakdown(texts)
        return scored["aggregate"]

    def score_with_breakdown(self, texts: Sequence[str]) -> dict[str, object]:
        clean_texts = [t for t in texts if t and t.strip()]
        if not clean_texts:
            return {
                # GHOST DATA PREVENTIE: 0.001 is leeg maar geïnitialiseerd. 0.0 = API crash.
                "aggregate": SentimentResult(score=0.001, confidence=0.0, model_name=self.model_name),
                "items": [],
            }

        results = self._pipe(clean_texts[:20], batch_size=8)
        signed_scores: list[float] = []
        confidences: list[float] = []
        details: list[dict[str, float | str]] = []
        for idx, row in enumerate(results):
            label = str(row.get("label", "")).lower()
            conf = float(row.get("score", 0.0))
            confidences.append(conf)
            if "positive" in label:
                signed = conf
            elif "negative" in label:
                signed = -conf
            else:
                signed = 0.0
            signed_scores.append(signed)
            details.append(
                {
                    "index": idx,
                    "label": label,
                    "confidence": round(conf, 4),
                    "signed_score": round(signed, 4),
                }
            )

        avg_score = sum(signed_scores) / max(1, len(signed_scores))
        avg_conf = sum(confidences) / max(1, len(confidences))
        aggregate = SentimentResult(
            score=round(max(-1.0, min(1.0, avg_score)), 4),
            confidence=round(avg_conf, 4),
            model_name=self.model_name,
        )
        return {"aggregate": aggregate, "items": details}


class LazyFinBertSentimentAnalyzer(SentimentAnalyzer):
    """Laadt FinBERT pas bij eerste sentiment-call (snellere app-import en eerste HTTP-responses)."""

    def __init__(self, model_id: str = "ProsusAI/finbert") -> None:
        self.model_id = model_id
        self.model_name = f"hf-{model_id}"
        self._inner: FinBertSentimentAnalyzer | None = None
        self._lock = threading.Lock()
        if str(os.getenv("FINBERT_EAGER_INIT", "0")).strip().lower() in ("1", "true", "yes", "on"):
            self._ensure()

    def _ensure(self) -> FinBertSentimentAnalyzer:
        with self._lock:
            if self._inner is None:
                self._inner = FinBertSentimentAnalyzer(model_id=self.model_id)
            return self._inner

    def score(self, texts: Sequence[str]) -> SentimentResult:
        return self._ensure().score(texts)

    def score_with_breakdown(self, texts: Sequence[str]) -> dict[str, object]:
        return self._ensure().score_with_breakdown(texts)


class RedisSentimentReader(SentimentAnalyzer):
    """
    Leest FinBERT-scores uit Redis (geschreven door de news-container).
    Valt terug op `fallback` als de Redis-key ontbreekt of verlopen is (TTL).
    Gebruikt threading.local voor thread-safe markt-context.
    """

    def __init__(self, redis_url: str, fallback: SentimentAnalyzer | None = None) -> None:
        self._redis_url = redis_url
        self._fallback = fallback
        self._local = threading.local()
        self._r: object | None = None
        self._r_lock = threading.Lock()

    def _redis(self):
        if self._r is None:
            with self._r_lock:
                if self._r is None:
                    try:
                        import redis as redis_lib
                        self._r = redis_lib.from_url(
                            self._redis_url,
                            decode_responses=True,
                            socket_connect_timeout=1,
                            socket_timeout=1,
                        )
                    except Exception:
                        self._r = None
        return self._r

    def set_market(self, market: str) -> None:
        """Stel de markt-context in voor de huidige thread (roep aan vóór score_with_breakdown)."""
        self._local.market = str(market or "").strip().upper().replace("/", "-")

    def score(self, texts: Sequence[str]) -> SentimentResult:
        return self.score_with_breakdown(texts)["aggregate"]

    def score_with_breakdown(self, texts: Sequence[str]) -> dict[str, object]:
        market = getattr(self._local, "market", "")
        if market:
            try:
                r = self._redis()
                if r is not None:
                    raw = r.get(f"news:sentiment:{market}")
                    if raw:
                        data = json.loads(raw)
                        agg = SentimentResult(
                            score=float(data.get("score", 0.0)),
                            confidence=float(data.get("confidence", 0.0)),
                            model_name="redis-cached-finbert",
                        )
                        return {"aggregate": agg, "items": data.get("items", [])}
            except Exception:
                pass
        if self._fallback is not None:
            return self._fallback.score_with_breakdown(texts)
        return {
            "aggregate": SentimentResult(score=0.0, confidence=0.0, model_name="neutral-fallback"),
            "items": [],
        }
