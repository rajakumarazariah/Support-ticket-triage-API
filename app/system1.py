"""System 1: the fast-path intent classifier.

Loads the artifact written by `scripts/train_system1.py` (calibrated logistic
regression, train embeddings, metadata) plus the sentence encoder, ONCE at startup.
`predict(text)` returns intent, category, calibrated confidence and the cosine
similarity to the nearest training example (a cheap out-of-distribution signal).

Embedding must match training exactly: raw text, no instruction prefix, L2-normalised.
`predict` is synchronous and CPU-bound; async callers should use `asyncio.to_thread`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np

from app.schemas import INTENT_TO_CATEGORY, Category, Intent

log = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], np.ndarray]
REQUIRED_METADATA = ("tau", "classes", "intent_to_category", "encoder", "embedding_dim")


class System1Error(RuntimeError):
    """Raised when the System 1 artifact is invalid or inconsistent with the schemas."""


@dataclass(frozen=True)
class System1Prediction:
    """Result of a System 1 prediction."""

    intent: Intent
    category: Category
    confidence: float  # calibrated top-class probability
    similarity: float  # cosine similarity to the nearest training example
    nn_intent: Intent | None  # intent of that nearest training example


def default_embed_fn(model_name: str) -> EmbedFn:
    """Load a SentenceTransformer and return an embed function (normalised float32)."""
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    return embed


class System1:
    """Calibrated intent classifier with nearest-neighbour similarity."""

    def __init__(
        self,
        classifier: Any,
        train_embeddings: np.ndarray,
        train_labels: Sequence[str],
        metadata: Mapping[str, Any],
        embed_fn: EmbedFn,
    ) -> None:
        self._validate(classifier, train_embeddings, train_labels, metadata)
        self._clf = classifier
        self._train = np.asarray(train_embeddings, dtype=np.float32)  # stored as float16
        self._labels = np.asarray(train_labels)
        self._meta = dict(metadata)
        self._embed = embed_fn
        self._classes = [str(c) for c in classifier.classes_]

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, path: Path, embed_fn: EmbedFn | None = None, warmup: bool = True) -> "System1":
        """Load the joblib artifact and the encoder; optionally run one warm-up prediction.

        Raises:
            System1Error: artifact missing keys or inconsistent with app.schemas.
            FileNotFoundError: artifact path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"System 1 model not found: {path}")
        artifact = joblib.load(path)
        for key in ("classifier", "train_embeddings", "train_labels", "metadata"):
            if key not in artifact:
                raise System1Error(f"artifact missing '{key}'")
        meta = artifact["metadata"]
        missing = [k for k in REQUIRED_METADATA if k not in meta]
        if missing:
            raise System1Error(f"artifact metadata missing {missing}")
        embed = embed_fn or default_embed_fn(meta["encoder"])
        system1 = cls(
            artifact["classifier"], artifact["train_embeddings"], artifact["train_labels"], meta, embed
        )
        if warmup:
            system1.predict("warm up")  # loads weights now and fails fast on a bad encoder
        log.info("System 1 loaded", extra={"path": str(path), "tau": system1.tau,
                                           "n_train": len(system1._labels)})
        return system1

    @staticmethod
    def _validate(clf: Any, emb: np.ndarray, labels: Sequence[str], meta: Mapping[str, Any]) -> None:
        missing = [k for k in REQUIRED_METADATA if k not in meta]
        if missing:
            raise System1Error(f"metadata missing {missing}")
        tau = meta["tau"]
        if not isinstance(tau, (int, float)) or not 0.0 < tau <= 1.0:
            raise System1Error(f"tau must be in (0, 1], got {tau!r}")
        valid = {i.value for i in Intent}
        unknown = sorted({str(c) for c in clf.classes_} - valid)
        if unknown:
            raise System1Error(f"classifier has intents not in app.schemas.Intent: {unknown}")
        for intent, category in meta["intent_to_category"].items():
            if intent in valid and INTENT_TO_CATEGORY[Intent(intent)].value != category:
                raise System1Error(
                    f"category mismatch for '{intent}': artifact says {category}, "
                    f"schemas say {INTENT_TO_CATEGORY[Intent(intent)].value}"
                )
        emb = np.asarray(emb)
        if emb.ndim != 2 or emb.shape[1] != meta["embedding_dim"]:
            raise System1Error(f"train_embeddings shape {emb.shape} != dim {meta['embedding_dim']}")
        if len(labels) != emb.shape[0]:
            raise System1Error("train_labels and train_embeddings have different lengths")

    # -- properties -----------------------------------------------------------
    @property
    def tau(self) -> float:
        """Confidence threshold chosen at training time."""
        return float(self._meta["tau"])

    @property
    def metadata(self) -> dict[str, Any]:
        """Copy of the artifact metadata."""
        return dict(self._meta)

    # -- inference ------------------------------------------------------------
    def predict(self, text: str) -> System1Prediction:
        """Predict intent, category, calibrated confidence and nearest-neighbour similarity."""
        vec = np.asarray(self._embed([text]), dtype=np.float32)
        if vec.ndim != 2 or vec.shape != (1, self._train.shape[1]):
            raise System1Error(f"encoder returned shape {vec.shape}, expected (1, {self._train.shape[1]})")
        q = vec[0]
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm  # defensive: training embeddings are unit length

        proba = self._clf.predict_proba(q[None, :])[0]
        top = int(np.argmax(proba))
        intent = Intent(self._classes[top])

        sims = self._train @ q
        nn = int(np.argmax(sims))
        nn_label = str(self._labels[nn])
        return System1Prediction(
            intent=intent,
            category=INTENT_TO_CATEGORY[intent],
            confidence=float(np.clip(proba[top], 0.0, 1.0)),
            similarity=float(np.clip(sims[nn], -1.0, 1.0)),
            nn_intent=Intent(nn_label) if nn_label in {i.value for i in Intent} else None,
        )