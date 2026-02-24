"""
Utility functions for loading, filtering, and exporting climate tweet data.
"""

import re
import json
import os

import pandas as pd

from config import DENIAL_KEYWORDS


def load_data(path: str) -> pd.DataFrame:
    """
    Reads a CSV and performs basic text cleaning.
    Required column: 'text'.
    """
    df = pd.read_csv(path, encoding="utf-8")

    if "text" not in df.columns:
        raise ValueError(
            f"CSV must contain a 'text' column. Found: {set(df.columns)}"
        )

    df["text"] = df["text"].astype(str).apply(_clean_text)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    print(f"  Loaded {len(df)} valid tweets from '{path}'")
    return df


def _clean_text(text: str) -> str:
    """Removes URLs, mentions, and normalizes whitespace."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_denial_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Filters tweets that contain at least one denial-related keyword."""
    pattern = "|".join(re.escape(kw) for kw in DENIAL_KEYWORDS)
    mask = df["text"].str.lower().str.contains(pattern, na=False)
    result = df[mask].copy().reset_index(drop=True)
    print(f"  Filtered: {len(result)} denial-related tweets (from {len(df)} total)")
    return result


def save_results(df: pd.DataFrame, name: str) -> str:
    """Exports the DataFrame to 'results/<name>.csv'."""
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Results saved: {path}")
    return path


def compute_stats(df: pd.DataFrame) -> dict:
    """Computes sentiment counts, percentages, and average confidence scores."""
    total = len(df)
    counts = df["sentiment"].value_counts().to_dict()

    stats = {
        "total_tweets": total,
        "counts": counts,
        "percentages": {
            s: round((c / total) * 100, 2) for s, c in counts.items()
        },
    }

    if "score" in df.columns:
        stats["avg_score"] = (
            df.groupby("sentiment")["score"]
            .mean()
            .round(4)
            .to_dict()
        )

    return stats


def export_stats_json(stats: dict, name: str) -> str:
    """Saves stats dict as JSON in 'results/'."""
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}_stats.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Stats exported: {path}")
    return path
