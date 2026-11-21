"""
EDA utilities — TV1 EXP-01.

Các hàm phân tích dữ liệu:
  - Phân phối nhãn
  - Thống kê độ dài văn bản
  - Từ phổ biến theo lớp
  - Kiểm tra duplicate, overlap cross-split
  - Phát hiện patterns đặc thù: URL, emoji, teencode markers

Dùng trong notebook: notebooks/EXP-01-EDA.ipynb
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # fallback safe font
import numpy as np
import pandas as pd
import seaborn as sns


LABEL_NAMES = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
LABEL_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_jsonl_to_df(path: str | Path) -> pd.DataFrame:
    """Load JSONL contract v4 thành DataFrame."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    return df


# ---------------------------------------------------------------------------
# 1. Phân phối nhãn
# ---------------------------------------------------------------------------

def plot_label_distribution(df: pd.DataFrame, title: str = "Phân phối nhãn") -> plt.Figure:
    """Vẽ bar chart phân phối nhãn theo split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall
    counts = df["label_name"].value_counts().reindex(["CLEAN", "OFFENSIVE", "HATE"])
    colors = [LABEL_COLORS[i] for i in range(3)]
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title(f"{title} — Tổng thể", fontsize=13)
    axes[0].set_ylabel("Số lượng")
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:,}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    # By split
    split_counts = df.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    split_counts = split_counts.reindex(columns=["CLEAN", "OFFENSIVE", "HATE"])
    split_counts.plot(
        kind="bar",
        ax=axes[1],
        color=colors,
        edgecolor="black",
        alpha=0.85,
        rot=0,
    )
    axes[1].set_title(f"{title} — Theo split", fontsize=13)
    axes[1].set_ylabel("Số lượng")
    axes[1].legend(title="Nhãn")

    plt.tight_layout()
    return fig


def print_label_stats(df: pd.DataFrame) -> None:
    """In bảng thống kê nhãn."""
    print("=== Phân phối nhãn ===")
    for split in df["split"].unique():
        sub = df[df["split"] == split]
        total = len(sub)
        print(f"\n  Split: {split} (n={total:,})")
        for label_id, label_name in LABEL_NAMES.items():
            cnt = (sub["label"] == label_id).sum()
            print(f"    {label_name:12s}: {cnt:5d} ({cnt/total*100:.1f}%)")


# ---------------------------------------------------------------------------
# 2. Thống kê độ dài văn bản
# ---------------------------------------------------------------------------

def compute_text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột word_count và char_count."""
    df = df.copy()
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()
    return df


def plot_text_length_distribution(df: pd.DataFrame) -> plt.Figure:
    """Boxplot độ dài theo nhãn."""
    if "word_count" not in df.columns:
        df = compute_text_lengths(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Word count
    data_by_label = [df[df["label"] == i]["word_count"].values for i in range(3)]
    axes[0].boxplot(
        data_by_label,
        tick_labels=["CLEAN", "OFFENSIVE", "HATE"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    axes[0].set_title("Phân phối số từ theo nhãn", fontsize=13)
    axes[0].set_ylabel("Số từ")
    axes[0].set_xlabel("Nhãn")

    # Char count
    data_by_label_char = [df[df["label"] == i]["char_count"].values for i in range(3)]
    axes[1].boxplot(
        data_by_label_char,
        tick_labels=["CLEAN", "OFFENSIVE", "HATE"],
        patch_artist=True,
        boxprops=dict(facecolor="lightyellow"),
    )
    axes[1].set_title("Phân phối số ký tự theo nhãn", fontsize=13)
    axes[1].set_ylabel("Số ký tự")
    axes[1].set_xlabel("Nhãn")

    plt.tight_layout()
    return fig


def print_length_stats(df: pd.DataFrame) -> None:
    """In thống kê mô tả độ dài."""
    if "word_count" not in df.columns:
        df = compute_text_lengths(df)
    print("=== Thống kê độ dài văn bản (word_count) ===")
    stats = df.groupby("label_name")["word_count"].describe()
    print(stats.to_string())
    print(f"\n  Tổng: mean={df['word_count'].mean():.1f}, "
          f"median={df['word_count'].median():.0f}, "
          f"max={df['word_count'].max()}")


# ---------------------------------------------------------------------------
# 3. Từ phổ biến theo lớp
# ---------------------------------------------------------------------------

def top_words_by_label(
    df: pd.DataFrame,
    label_id: int,
    top_n: int = 20,
    split: str = "train",
) -> list[tuple[str, int]]:
    """Trả list (word, count) phổ biến nhất trong class label_id."""
    sub = df[(df["label"] == label_id) & (df["split"] == split)]
    counter: Counter = Counter()
    for text in sub["text"]:
        tokens = str(text).lower().split()
        counter.update(tokens)
    return counter.most_common(top_n)


def plot_top_words(df: pd.DataFrame, top_n: int = 15, split: str = "train") -> plt.Figure:
    """Bar chart top words cho mỗi nhãn."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (label_id, label_name) in enumerate(LABEL_NAMES.items()):
        top = top_words_by_label(df, label_id, top_n=top_n, split=split)
        if not top:
            continue
        words, counts = zip(*top)
        color = LABEL_COLORS[label_id]
        axes[i].barh(list(reversed(words)), list(reversed(counts)), color=color, alpha=0.85)
        axes[i].set_title(f"Top {top_n} từ — {label_name}", fontsize=12)
        axes[i].set_xlabel("Tần suất")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Phát hiện patterns đặc thù
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
_HTML_RE = re.compile(r"<[^>]+>")
_NUMBER_RE = re.compile(r"\b\d+\b")


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các cột boolean phát hiện pattern."""
    df = df.copy()
    df["has_url"] = df["text"].str.contains(_URL_RE).fillna(False)
    df["has_emoji"] = df["text"].str.contains(_EMOJI_RE).fillna(False)
    df["has_html"] = df["text"].str.contains(_HTML_RE).fillna(False)
    df["has_number"] = df["text"].str.contains(_NUMBER_RE).fillna(False)
    # Teencode proxy: từ ngắn 1-2 ký tự không phải stopword
    df["has_short_tokens"] = df["text"].apply(
        lambda t: any(len(tok) <= 2 and tok.isalpha() for tok in str(t).split())
    )
    return df


def print_pattern_stats(df: pd.DataFrame) -> None:
    """In tỷ lệ pattern theo nhãn (chỉ split=train)."""
    if "has_url" not in df.columns:
        df = detect_patterns(df)
    train = df[df["split"] == "train"]
    patterns = ["has_url", "has_emoji", "has_html", "has_number", "has_short_tokens"]
    print("=== Tỷ lệ patterns đặc thù (train) ===")
    header = f"{'Pattern':<20} " + "  ".join(f"{n:>12}" for n in LABEL_NAMES.values())
    print(header)
    for pat in patterns:
        row = f"{pat:<20} "
        for label_id in LABEL_NAMES:
            sub = train[train["label"] == label_id]
            rate = sub[pat].mean() * 100 if len(sub) > 0 else 0
            row += f"  {rate:>11.1f}%"
        print(row)


# ---------------------------------------------------------------------------
# 5. Duplicate & cross-split overlap
# ---------------------------------------------------------------------------

def check_duplicates(df: pd.DataFrame) -> dict:
    """Kiểm tra exact duplicate và cross-split text overlap."""
    exact_dupes = df["text"].duplicated(keep=False).sum()
    exact_dupe_rows = df[df["text"].duplicated(keep=False)]

    # Cross-split overlap: cùng text, khác split
    train_texts = set(df[df["split"] == "train"]["text"])
    val_texts = set(df[df["split"] == "val"]["text"])
    overlap = train_texts & val_texts

    print("=== Kiểm tra duplicate ===")
    print(f"  Tổng records: {len(df):,}")
    print(f"  Exact duplicates (cùng text, bất kỳ split): {exact_dupes:,}")
    print(f"  Cross-split text overlap (train ∩ val): {len(overlap):,}")

    if len(overlap) > 0:
        print(f"  WARNING: Có {len(overlap)} text xuất hiện ở cả train lẫn val!")
        print("  Ví dụ:")
        for t in list(overlap)[:3]:
            print(f"    - {t[:80]!r}")
    else:
        print("  OK: Không có text overlap giữa train và val.")

    return {
        "total_records": len(df),
        "exact_duplicates": int(exact_dupes),
        "cross_split_overlap": len(overlap),
        "overlap_examples": list(overlap)[:5],
    }


# ---------------------------------------------------------------------------
# 6. Confusion matrix visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: list[list[int]],
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> plt.Figure:
    """Vẽ confusion matrix đẹp từ list 3x3."""
    cm_arr = np.array(cm)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm_arr / row_sums, 0)
        fmt_str = ".2%"
    else:
        cm_plot = cm_arr
        fmt_str = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt_str,
        xticklabels=list(LABEL_NAMES.values()),
        yticklabels=list(LABEL_NAMES.values()),
        cmap="Blues",
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Per-class F1 comparison (nhiều variants)
# ---------------------------------------------------------------------------

def plot_f1_comparison(
    results: list[dict],
    title: str = "So sánh F1 các biến thể EXP-01",
) -> plt.Figure:
    """
    results: list of {variant, metrics: {macro_f1, per_class: {CLEAN: {f1}, ...}}}
    """
    labels_order = ["CLEAN", "OFFENSIVE", "HATE"]
    n_variants = len(results)
    x = np.arange(len(labels_order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(results):
        f1s = [r["metrics"]["per_class"][l]["f1"] for l in labels_order]
        offset = (i - n_variants / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1s, width, label=f"EXP-01-{r['variant']}", alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_order)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig
