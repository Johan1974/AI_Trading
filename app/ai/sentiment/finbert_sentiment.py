"""
BESTANDSNAAM: /home/johan/AI_Trading/app/ai/sentiment/finbert_sentiment.py
FUNCTIE: Sentiment analyzer met HuggingFace Transformers FinBERT.
"""

import os
import threading
from collections.abc import Sequence

import torch
from transformers import pipeline

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
