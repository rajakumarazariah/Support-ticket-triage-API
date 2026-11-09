"""Train the System 1 fast-path intent classifier.

Embeds ticket text with bge-small-en-v1.5, fits a calibrated logistic regression,
evaluates on the dev split (accuracy, macro-F1, ECE), saves a reliability plot and
an accuracy-vs-coverage curve, and picks the confidence threshold tau: the lowest
confidence at which dev accuracy among accepted predictions meets a target.

Calibration is fitted with cross-validation on TRAIN only, so dev stays untouched
for evaluation and tau selection. Report final numbers on golden_test later.

Usage:
    python scripts/train_system1.py --target-accuracy 0.97
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"

EmbedFn = Callable[[Sequence[str]], np.ndarray]


# ----------------------------------------------------------------------------- IO
def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_encoder(model_name: str = DEFAULT_ENCODER, batch_size: int = 64) -> EmbedFn:
    """Return a function mapping texts to L2-normalised float32 embeddings."""
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    return embed


# ------------------------------------------------------------------------ metrics
def reliability_bins(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean confidence, accuracy, count) over equal-width bins in [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1])  # 0..n_bins-1
    mean_conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        count[b] = int(mask.sum())
        if count[b]:
            mean_conf[b] = conf[mask].mean()
            acc[b] = correct[mask].mean()
    return mean_conf, acc, count


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE: weighted mean |accuracy - confidence| across confidence bins."""
    mean_conf, acc, count = reliability_bins(conf, correct, n_bins)
    total = count.sum()
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(acc - mean_conf) * count) / total)


def coverage_curve(
    conf: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each unique confidence value t (descending), return (t, coverage, accuracy)
    of the predictions accepted when the threshold is t (i.e. conf >= t)."""
    thresholds = np.unique(conf)[::-1]
    coverage = np.empty(len(thresholds))
    accuracy = np.empty(len(thresholds))
    n = len(conf)
    for i, t in enumerate(thresholds):
        mask = conf >= t
        coverage[i] = mask.sum() / n
        accuracy[i] = correct[mask].mean()
    return thresholds, coverage, accuracy


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target_accuracy: float) -> float:
    """Lowest threshold (max coverage) whose accepted set has accuracy >= target.

    Raises:
        ValueError: if no threshold reaches the target accuracy.
    """
    thresholds, _, accuracy = coverage_curve(conf, correct)
    ok = np.where(accuracy >= target_accuracy)[0]
    if len(ok) == 0:
        raise ValueError(
            f"No threshold reaches {target_accuracy:.3f} accuracy "
            f"(best is {accuracy.max():.3f}). Lower the target or improve the model."
        )
    return float(thresholds[ok.max()])  # thresholds are descending, so max index = lowest t


# ------------------------------------------------------------------------- plots
def plot_reliability(conf: np.ndarray, correct: np.ndarray, path: Path, n_bins: int = 10) -> None:
    """Save a reliability diagram (confidence vs accuracy per bin)."""
    mean_conf, acc, count = reliability_bins(conf, correct, n_bins)
    keep = count > 0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(mean_conf[keep], acc[keep], "o-", label="System 1 (dev)")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram (ECE={expected_calibration_error(conf, correct, n_bins):.4f})")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage(
    conf: np.ndarray, correct: np.ndarray, tau: float, target: float, path: Path
) -> None:
    """Save accuracy-vs-coverage curve with the chosen operating point marked."""
    _, coverage, accuracy = coverage_curve(conf, correct)
    mask = conf >= tau
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverage, accuracy, "-")
    ax.axhline(target, ls="--", color="gray", label=f"target accuracy {target:.2f}")
    ax.plot(
        mask.mean(), correct[mask].mean(), "ro", label=f"tau={tau:.3f} (coverage {mask.mean():.1%})"
    )
    ax.set_xlabel("Coverage (fraction handled by System 1)")
    ax.set_ylabel("Accuracy on accepted tickets")
    ax.set_title("Accuracy vs coverage (dev)")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ training
def run_training(
    train_x: np.ndarray,
    train_y: Sequence[str],
    dev_x: np.ndarray,
    dev_y: Sequence[str],
    intent_to_category: dict[str, str],
    *,
    target_accuracy: float,
    plots_dir: Path,
    calibration: str = "sigmoid",
    cv: int = 5,
    C: float = 10.0,
    seed: int = 42,
    encoder_name: str = DEFAULT_ENCODER,
) -> tuple[dict, dict]:
    """Fit, evaluate, choose tau, and return (artifact, metrics).

    The artifact holds the calibrated classifier, the train embeddings/labels (for
    nearest-neighbour similarity in the router) and metadata including tau.
    """
    base = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf = CalibratedClassifierCV(base, method=calibration, cv=cv)
    clf.fit(train_x, train_y)

    proba = clf.predict_proba(dev_x)
    pred = clf.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    dev_y_arr = np.asarray(dev_y)
    correct = (pred == dev_y_arr).astype(float)

    tau = choose_threshold(conf, correct, target_accuracy)
    accepted = conf >= tau
    metrics = {
        "dev_accuracy": float(accuracy_score(dev_y_arr, pred)),
        "dev_macro_f1": float(f1_score(dev_y_arr, pred, average="macro")),
        "dev_ece": expected_calibration_error(conf, correct),
        "tau": tau,
        "target_accuracy": target_accuracy,
        "dev_coverage_at_tau": float(accepted.mean()),
        "dev_accuracy_at_tau": float(correct[accepted].mean()),
        "n_train": len(train_y),
        "n_dev": len(dev_y),
    }

    plot_reliability(conf, correct, plots_dir / "system1_reliability.png")
    plot_coverage(conf, correct, tau, target_accuracy, plots_dir / "system1_coverage.png")

    artifact = {
        "classifier": clf,
        "train_embeddings": train_x.astype(np.float16),  # normalised; cast back at load
        "train_labels": np.asarray(train_y),
        "metadata": {
            "tau": tau,
            "encoder": encoder_name,
            "embedding_dim": int(train_x.shape[1]),
            "classes": [str(c) for c in clf.classes_],
            "intent_to_category": intent_to_category,
            "calibration": calibration,
            "C": C,
            "seed": seed,
            "metrics": metrics,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return artifact, metrics


def main(argv: list[str] | None = None, embed_fn: EmbedFn | None = None) -> None:
    """CLI entry point. `embed_fn` can be injected for testing."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--dev", type=Path, default=Path("data/golden_dev.jsonl"))
    p.add_argument("--out", type=Path, default=Path("models/system1_v1.joblib"))
    p.add_argument("--plots-dir", type=Path, default=Path("results"))
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--target-accuracy", type=float, default=0.97)
    p.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    train, dev = load_jsonl(args.train), load_jsonl(args.dev)
    embed = embed_fn or make_encoder(args.encoder)
    train_x = embed([r["text"] for r in train])
    dev_x = embed([r["text"] for r in dev])

    intent_to_category = {r["intent"]: r["category"] for r in train + dev}
    artifact, metrics = run_training(
        train_x,
        [r["intent"] for r in train],
        dev_x,
        [r["intent"] for r in dev],
        intent_to_category,
        target_accuracy=args.target_accuracy,
        plots_dir=args.plots_dir,
        calibration=args.calibration,
        cv=args.cv,
        C=args.C,
        seed=args.seed,
        encoder_name=args.encoder,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out, compress=3)

    print("\n=== System 1 (dev) ===")
    print(f"accuracy            {metrics['dev_accuracy']:.4f}")
    print(f"macro-F1            {metrics['dev_macro_f1']:.4f}")
    print(f"ECE                 {metrics['dev_ece']:.4f}")
    print(f"tau                 {metrics['tau']:.4f}  (target acc {args.target_accuracy:.2f})")
    print(f"coverage @ tau      {metrics['dev_coverage_at_tau']:.1%}")
    print(f"accuracy @ tau      {metrics['dev_accuracy_at_tau']:.4f}")
    print(f"saved model         {args.out}")
    print(f"saved plots         {args.plots_dir}/system1_reliability.png, system1_coverage.png")


if __name__ == "__main__":
    main()