"""Make the repo root importable so tests can `import app...`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---- shared System 1 fixture (synthetic embeddings, no model download) -------
import zlib  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

S1_INTENTS = ["track_order", "get_refund", "cancel_order", "payment_issue"]


def s1_fake_embed(texts):
    """Class-clustered unit vectors. 'ood ...' texts live on an axis no class uses."""
    out = []
    for t in texts:
        head = t.split()[0]
        rng = np.random.default_rng(zlib.crc32(t.encode()))
        center = np.zeros(16)
        if head == "ood":
            center[10] = 1.0
        elif head in S1_INTENTS:
            center[S1_INTENTS.index(head)] = 1.0
        else:
            center[0] = 1.0  # e.g. the "warm up" call
        v = center + rng.normal(scale=0.12, size=16)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


@pytest.fixture(scope="session")
def system1_model(tmp_path_factory):
    """Train a tiny System 1 on synthetic data; return (joblib_path, embed_fn)."""
    import joblib

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_system1 as ts

    from app.schemas import INTENT_TO_CATEGORY, Intent

    def rows(tag, n):
        texts = [f"{i} sample {tag} {k}" for i in S1_INTENTS for k in range(n)]
        labels = [i for i in S1_INTENTS for _ in range(n)]
        return texts, labels

    tr_t, tr_y = rows("tr", 60)
    dv_t, dv_y = rows("dv", 40)
    mapping = {i: INTENT_TO_CATEGORY[Intent(i)].value for i in S1_INTENTS}
    d = tmp_path_factory.mktemp("s1")
    artifact, _ = ts.run_training(
        s1_fake_embed(tr_t), tr_y, s1_fake_embed(dv_t), dv_y, mapping,
        target_accuracy=0.9, plots_dir=d, cv=3,
    )
    path = d / "system1.joblib"
    joblib.dump(artifact, path)
    return path, s1_fake_embed