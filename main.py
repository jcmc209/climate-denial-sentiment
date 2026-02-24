"""
Sentiment Analysis of Climate Change Denialism on Twitter.
Model: cardiffnlp/twitter-roberta-base-sentiment-latest

Usage:
    python main.py --input data/tweets_combinado.csv
    python main.py --input data/tweets_combinado.csv --no-charts
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import MODEL_NAME
from utils import (
    load_data,
    filter_denial_tweets,
    save_results,
    compute_stats,
    export_stats_json,
)


# ---------------------------------------------------------------------------
# SENTIMENT MODEL
# ---------------------------------------------------------------------------

def load_model():
    """Loads the CardiffNLP Twitter-RoBERTa sentiment model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    print(f"  Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True,
    )
    print("  Model loaded successfully.")
    return pipe


LABEL_MAP = {
    "positive": "POS",
    "negative": "NEG",
    "neutral": "NEU",
}


def analyze_sentiment(df: pd.DataFrame, pipe) -> pd.DataFrame:
    """Classifies each tweet as POS / NEG / NEU with a confidence score."""
    sentiments = []
    scores = []

    for text in tqdm(df["text"], desc="  Analyzing", unit="tweet"):
        try:
            result = pipe(text[:512])[0]
            label = LABEL_MAP.get(result["label"].lower(), result["label"])
            sentiments.append(label)
            scores.append(round(result["score"], 4))
        except Exception:
            sentiments.append("NEU")
            scores.append(0.0)

    df = df.copy()
    df["sentiment"] = sentiments
    df["score"] = scores
    return df


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------

SENTIMENT_COLORS = {"POS": "#4CAF50", "NEG": "#F44336", "NEU": "#2196F3"}
CHART_DIR = os.path.join("results", "charts")


def _ensure_chart_dir():
    os.makedirs(CHART_DIR, exist_ok=True)


def chart_bars(df: pd.DataFrame):
    """Bar chart: tweet count by sentiment."""
    import matplotlib.pyplot as plt

    _ensure_chart_dir()
    counts = df["sentiment"].value_counts().reindex(["POS", "NEU", "NEG"], fill_value=0)
    colors = [SENTIMENT_COLORS[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.max() * 0.01, 5),
            f"{val:,}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Sentiment Distribution - Climate Denialism Tweets", fontsize=13, pad=12)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of tweets")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    path = os.path.join(CHART_DIR, "sentiment_bars.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


def chart_pie(df: pd.DataFrame):
    """Donut chart: sentiment proportions."""
    import matplotlib.pyplot as plt

    _ensure_chart_dir()
    counts = df["sentiment"].value_counts().reindex(["POS", "NEU", "NEG"], fill_value=0)
    colors = [SENTIMENT_COLORS[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
        textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontweight("bold")

    centre = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_artist(centre)
    ax.set_title("Sentiment Proportions - Climate Denialism Tweets", fontsize=13, pad=12)

    path = os.path.join(CHART_DIR, "sentiment_pie.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


def chart_confidence(df: pd.DataFrame):
    """Histogram of model confidence scores, split by sentiment."""
    import matplotlib.pyplot as plt

    _ensure_chart_dir()
    fig, ax = plt.subplots(figsize=(9, 5))

    for s in ["POS", "NEU", "NEG"]:
        subset = df[df["sentiment"] == s]["score"]
        if not subset.empty:
            ax.hist(subset, bins=30, alpha=0.6, label=s,
                    color=SENTIMENT_COLORS[s], edgecolor="white")

    ax.set_title("Confidence Score Distribution by Sentiment", fontsize=13, pad=12)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Number of tweets")
    ax.legend(title="Sentiment")
    ax.spines[["top", "right"]].set_visible(False)

    path = os.path.join(CHART_DIR, "confidence_histogram.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


def chart_wordcloud(df: pd.DataFrame):
    """Word cloud from tweet texts."""
    try:
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib.pyplot as plt
    except ImportError:
        print("  wordcloud not installed; skipping.")
        return

    _ensure_chart_dir()
    stopwords = set(STOPWORDS) | {
        "rt", "amp", "will", "one", "now", "just", "like", "get",
        "us", "also", "would", "could", "still", "even", "much",
        "say", "said", "new", "make", "going", "need", "people",
        "via", "want", "really", "think", "know", "see", "go",
        "climate", "change", "global", "warming",
    }

    full_text = " ".join(df["text"].tolist())
    if not full_text.strip():
        return

    wc = WordCloud(
        width=1000, height=500,
        background_color="white",
        stopwords=stopwords,
        colormap="RdYlGn",
        max_words=120,
        collocations=False,
    ).generate(full_text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud - Climate Denialism Tweets", fontsize=14, pad=12)

    path = os.path.join(CHART_DIR, "wordcloud.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


def chart_top_keywords(df: pd.DataFrame):
    """Bar chart of the most frequent words in denial tweets."""
    import matplotlib.pyplot as plt
    from collections import Counter
    import re

    _ensure_chart_dir()

    skip = {
        "climate", "change", "global", "warming", "that", "this", "with",
        "have", "from", "they", "will", "been", "their", "were", "would",
        "about", "what", "more", "when", "just", "like", "your", "some",
        "them", "than", "into", "could", "also", "other", "very", "even",
        "most", "only", "make", "over", "such", "then", "know", "much",
        "well", "going", "being", "https", "does", "doesn",
    }

    all_words = []
    for text in df["text"]:
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        all_words.extend(w for w in words if w not in skip)

    top = Counter(all_words).most_common(20)
    words_list = [w for w, _ in top]
    counts_list = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(words_list)), counts_list, color="#607D8B", edgecolor="white")
    ax.set_yticks(range(len(words_list)))
    ax.set_yticklabels(words_list, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Most Frequent Words in Denialism Tweets", fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, counts_list):
        ax.text(bar.get_width() + max(counts_list) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, fontweight="bold")

    path = os.path.join(CHART_DIR, "top_keywords.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


def save_top_examples(df: pd.DataFrame):
    """Saves the top-10 most confident tweets per sentiment to a text file."""
    _ensure_chart_dir()
    lines = []
    for s in ["POS", "NEU", "NEG"]:
        subset = df[df["sentiment"] == s].nlargest(10, "score")
        lines.append(f"\n{'='*70}")
        lines.append(f"  TOP 10 MOST CONFIDENT {s} TWEETS")
        lines.append(f"{'='*70}")
        for _, row in subset.iterrows():
            lines.append(f"  [{row['score']:.4f}] {row['text'][:140]}")

    path = os.path.join(CHART_DIR, "top_examples.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Top examples saved: {path}")


def generate_all_charts(df: pd.DataFrame):
    """Generates the full set of visualizations."""
    chart_bars(df)
    chart_pie(df)
    chart_confidence(df)
    chart_wordcloud(df)
    chart_top_keywords(df)
    save_top_examples(df)


# ---------------------------------------------------------------------------
# CONSOLE OUTPUT
# ---------------------------------------------------------------------------

def print_summary(stats: dict):
    """Prints a formatted summary to the console."""
    sep = "-" * 55
    print(f"\n{sep}")
    print("  RESULTS - Climate Change Denialism")
    print(sep)
    print(f"  Total tweets analyzed: {stats['total_tweets']:,}")
    print()
    for s in ["POS", "NEU", "NEG"]:
        count = stats["counts"].get(s, 0)
        pct = stats["percentages"].get(s, 0.0)
        bar = "#" * int(pct / 2)
        print(f"  {s}  {bar:<50}  {count:>6,} tweets  ({pct:.1f}%)")

    if "avg_score" in stats:
        print()
        print("  Average confidence:")
        for s in ["POS", "NEU", "NEG"]:
            score = stats["avg_score"].get(s, 0)
            print(f"    {s}: {score:.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis of Climate Change Denialism on Twitter."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV (must have a 'text' column).",
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation.",
    )
    args = parser.parse_args()

    # 1. Load data
    print("\n[1/4] Loading data...")
    df = load_data(args.input)

    # 2. Filter denial tweets
    print("\n[2/4] Filtering denial-related tweets...")
    df = filter_denial_tweets(df)

    if df.empty:
        print("  ERROR: No denial-related tweets found. Check your keywords.")
        return

    # 3. Load model and analyze
    print("\n[3/4] Loading model and analyzing sentiment...")
    pipe = load_model()
    df = analyze_sentiment(df, pipe)

    # 4. Export results
    print("\n[4/4] Exporting results...")
    save_results(df, "denial_sentiment")

    stats = compute_stats(df)
    export_stats_json(stats, "denial_sentiment")
    print_summary(stats)

    if not args.no_charts:
        print("\n  Generating charts...")
        generate_all_charts(df)

    print(f"\n  Done. All output files in 'results/'.")


if __name__ == "__main__":
    main()
