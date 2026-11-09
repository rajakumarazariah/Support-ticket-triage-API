from __future__ import annotations
import argparse
import json
import re 
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ID  = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_COLUMNS = ["id", "text", "category", "intent", "reference_reply"]
SPLIT_FILES = {"train":"train.json1", "dev": "golden_dev.json1", "test":"golden_test.json1"}

def load_raw(dataset_is : str = DATASET_ID) -> pd.DataFrame:
    from datasets import load_dataset
    ds= load_dataset(DATASET_ID,split="train")
    return ds.to_pandas()

def normalise(text:str) ->str:
    return re.sub(r"\s+","",str(text)).strip()

def clean(df:pd.DataFrame) -> pd.DataFrame:
    out = df[["instruction", "category", "intent", "response"]].copy()
    out["instruction"] = out["instruction"].map(normalise)
    out["response"]= out["response"].map(normalise)
    out = out[out["instruction"]!=""]

    key = out["instruction"].str.lower()
    n_intents= out.groupby(key)["intent"].transform("nunique")
    conflicting = n_intents > 1
    if conflicting.any():
        print(f"Dropping {int(conflicting.sum())}rows with conflicting intents")
    out= out[~conflicting]
    before = len(out)
    out = out.loc[~out["instruction"].str.lower().duplicated(keep="first")]
    print(f"Dropped {before - len(out)} duplicate instructions; {len(out)} rows remain")
    return out.reset_index(drop=True)

def check_intent_category(df: pd.DataFrame) -> dict[str, str]:
    """Verify each intent maps to exactly one category; return the mapping.
 
    Raises:
        ValueError: if any intent appears under more than one category.
    """
    counts = df.groupby("intent")["category"].nunique()
    bad = counts[counts > 1]
    if not bad.empty:
        raise ValueError(f"Intents mapped to multiple categories: {bad.to_dict()}")
    return df.groupby("intent")["category"].first().to_dict()
 
 
def make_splits(
    df: pd.DataFrame, seed: int = 42, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> dict[str, pd.DataFrame]:
    """Stratified-by-intent train/dev/test split.
 
    Raises:
        ValueError: if ratios don't sum to 1 or an intent has too few rows to stratify.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    min_count = df["intent"].value_counts().min()
    if min_count < 3:
        raise ValueError(f"Need >=3 rows per intent to stratify 3 ways; min is {min_count}")
 
    train_r, dev_r, test_r = ratios
    train, rest = train_test_split(
        df, test_size=dev_r + test_r, stratify=df["intent"], random_state=seed
    )
    dev, test = train_test_split(
        rest, test_size=test_r / (dev_r + test_r), stratify=rest["intent"], random_state=seed
    )
    return {"train": train, "dev": dev, "test": test}
 
 
def to_records(df: pd.DataFrame, prefix: str) -> list[dict]:
    """Convert a split to output records with stable ids like `dev-00012`."""
    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        records.append(
            {
                "id": f"{prefix}-{i:05d}",
                "text": row.instruction,
                "category": row.category,
                "intent": row.intent,
                "reference_reply": row.response,
            }
        )
    return records
 
 
def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records as one JSON object per line (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
 
 
def print_class_counts(splits: dict[str, pd.DataFrame]) -> None:
    """Print per-intent counts for each split side by side."""
    table = pd.DataFrame({name: part["intent"].value_counts() for name, part in splits.items()})
    table = table.fillna(0).astype(int).sort_index()
    table["total"] = table.sum(axis=1)
    print("\nClass counts per intent:")
    print(table.to_string())
    print("\nSplit sizes:", {name: len(part) for name, part in splits.items()})
 
 
def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
 
    df = clean(load_raw())
    mapping = check_intent_category(df)
    print(f"{len(mapping)} intents, {len(set(mapping.values()))} categories; mapping is consistent")
 
    splits = make_splits(df, seed=args.seed)
    for name, part in splits.items():
        write_jsonl(to_records(part, name), args.out_dir / SPLIT_FILES[name])
    print_class_counts(splits)
 
 
if __name__ == "__main__":
    main()