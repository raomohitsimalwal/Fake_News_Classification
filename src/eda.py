"""
Exploratory Data Analysis (EDA) Module for Fake News and Claim Classification.
Generates comprehensive statistical summaries and high-resolution plots for:
1. Class distributions (LIAR 6-class, 3-class, and News Real/Fake).
2. Word count and character count distributions.
3. Top n-grams (unigrams & bigrams) contrasting credible vs deceptive claims.
4. Word clouds with distinct color palettes.
5. Speaker credibility patterns in the fact-checking dataset.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Style configuration for presentation-grade figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")


def generate_eda():
    os.makedirs(VIS_DIR, exist_ok=True)
    print("=" * 60)
    print("Executing Exploratory Data Analysis & Visualization Generation")
    print("=" * 60)

    # 1. Load Clean Datasets
    liar_path = os.path.join(DATA_DIR, "liar_processed.csv")
    news_path = os.path.join(DATA_DIR, "news_processed.csv")

    df_liar = pd.read_csv(liar_path)
    df_news = pd.read_csv(news_path)

    print(f"Loaded LIAR dataset: {df_liar.shape[0]} records.")
    print(f"Loaded News dataset: {df_news.shape[0]} records.")

    # ----------------------------------------------------
    # Plot 1: Class Distributions
    # ----------------------------------------------------
    print("\n[1/5] Generating Class Distribution Visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # LIAR 6-Class
    order_6class = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
    colors_6 = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]
    sns.countplot(
        data=df_liar,
        x="label",
        order=order_6class,
        palette=colors_6,
        ax=axes[0],
        hue="label",
        legend=False
    )
    axes[0].set_title("LIAR Dataset: 6 Credibility Classes", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("PolitiFact Credibility Rating", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    # LIAR 3-Class
    order_3class = ["Credible", "Half-True", "Questionable"]
    colors_3 = ["#2ecc71", "#f39c12", "#e74c3c"]
    sns.countplot(
        data=df_liar,
        x="credibility_3class",
        order=order_3class,
        palette=colors_3,
        ax=axes[1],
        hue="credibility_3class",
        legend=False
    )
    axes[1].set_title("LIAR Dataset: 3-Tier Credibility Grouping", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Credibility Tier", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)

    # News Real vs Fake
    order_news = ["REAL", "FAKE"]
    colors_news = ["#3498db", "#e74c3c"]
    sns.countplot(
        data=df_news,
        x="label",
        order=order_news,
        palette=colors_news,
        ax=axes[2],
        hue="label",
        legend=False
    )
    axes[2].set_title("News Dataset: Real vs. Fake Articles", fontsize=13, fontweight="bold", pad=12)
    axes[2].set_xlabel("Article Veracity", fontsize=11)
    axes[2].set_ylabel("Count", fontsize=11)

    for ax in axes:
        for p in ax.patches:
            val = int(p.get_height())
            ax.annotate(f"{val:,}", (p.get_x() + p.get_width() / 2., val),
                        ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                        textcoords="offset points")

    plt.tight_layout()
    plot1_path = os.path.join(VIS_DIR, "class_distribution.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot1_path}")

    # ----------------------------------------------------
    # Plot 2: Text Length Distribution (Word & Char Counts)
    # ----------------------------------------------------
    print("\n[2/5] Generating Text Length Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.boxplot(
        data=df_liar,
        x="credibility_3class",
        y="word_count",
        order=["Credible", "Half-True", "Questionable"],
        palette=["#2ecc71", "#f39c12", "#e74c3c"],
        showfliers=False,
        ax=axes[0],
        hue="credibility_3class",
        legend=False
    )
    axes[0].set_title("Claim Statement Word Count by Credibility Tier", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Credibility Tier", fontsize=11)
    axes[0].set_ylabel("Word Count (without outliers)", fontsize=11)

    sns.boxplot(
        data=df_news,
        x="label",
        y="word_count",
        palette=["#3498db", "#e74c3c"],
        showfliers=False,
        ax=axes[1],
        hue="label",
        legend=False
    )
    axes[1].set_title("News Article Word Count by Label", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("News Label", fontsize=11)
    axes[1].set_ylabel("Word Count (without outliers)", fontsize=11)

    plt.tight_layout()
    plot2_path = os.path.join(VIS_DIR, "text_length_distribution.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot2_path}")

    # ----------------------------------------------------
    # Plot 3: Top N-grams Comparison (Credible vs Questionable)
    # ----------------------------------------------------
    print("\n[3/5] Generating Top N-grams Comparison...")
    def get_top_ngrams(corpus, n=1, top_k=15):
        vec = CountVectorizer(ngram_range=(n, n), stop_words="english", max_features=1000)
        bag = vec.fit_transform(corpus.dropna())
        sum_words = bag.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]

    credible_claims = df_liar[df_liar["credibility_3class"] == "Credible"]["clean_statement"]
    questionable_claims = df_liar[df_liar["credibility_3class"] == "Questionable"]["clean_statement"]

    top_cred_bi = get_top_ngrams(credible_claims, n=2, top_k=10)
    top_quest_bi = get_top_ngrams(questionable_claims, n=2, top_k=10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    words_c, counts_c = zip(*top_cred_bi)
    axes[0].barh(range(len(words_c)), counts_c, color="#2ecc71", edgecolor="#27ae60")
    axes[0].set_yticks(range(len(words_c)))
    axes[0].set_yticklabels(words_c, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_title("Top 10 Bigrams in Credible Claims", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Frequency", fontsize=11)

    words_q, counts_q = zip(*top_quest_bi)
    axes[1].barh(range(len(words_q)), counts_q, color="#e74c3c", edgecolor="#c0392b")
    axes[1].set_yticks(range(len(words_q)))
    axes[1].set_yticklabels(words_q, fontsize=11)
    axes[1].invert_yaxis()
    axes[1].set_title("Top 10 Bigrams in Questionable / False Claims", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frequency", fontsize=11)

    plt.tight_layout()
    plot3_path = os.path.join(VIS_DIR, "top_ngrams_comparison.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot3_path}")

    # ----------------------------------------------------
    # Plot 4: Word Clouds (Credible vs Deceptive)
    # ----------------------------------------------------
    print("\n[4/5] Generating Word Clouds...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    wc_cred = WordCloud(
        width=800, height=450, background_color="white", colormap="summer",
        max_words=120, collocations=False, random_state=42
    ).generate(" ".join(credible_claims.dropna()))

    wc_quest = WordCloud(
        width=800, height=450, background_color="white", colormap="inferno",
        max_words=120, collocations=False, random_state=42
    ).generate(" ".join(questionable_claims.dropna()))

    axes[0].imshow(wc_cred, interpolation="bilinear")
    axes[0].set_title("Word Cloud: Lexicon of Credible Claims", fontsize=14, fontweight="bold", pad=12)
    axes[0].axis("off")

    axes[1].imshow(wc_quest, interpolation="bilinear")
    axes[1].set_title("Word Cloud: Lexicon of Questionable / False Claims", fontsize=14, fontweight="bold", pad=12)
    axes[1].axis("off")

    plt.tight_layout()
    plot4_path = os.path.join(VIS_DIR, "wordcloud_credible_vs_fake.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot4_path}")

    # ----------------------------------------------------
    # Plot 5: Speaker Credibility Analysis (LIAR dataset)
    # ----------------------------------------------------
    print("\n[5/5] Generating Speaker Credibility Breakdown...")
    top_speakers = df_liar["speaker"].value_counts().head(8).index.tolist()
    df_top_speakers = df_liar[df_liar["speaker"].isin(top_speakers)]

    speaker_cred = pd.crosstab(
        df_top_speakers["speaker"],
        df_top_speakers["credibility_3class"],
        normalize="index"
    ) * 100
    speaker_cred = speaker_cred[["Credible", "Half-True", "Questionable"]].sort_values("Questionable")

    fig, ax = plt.subplots(figsize=(12, 6))
    speaker_cred.plot(
        kind="barh",
        stacked=True,
        color=["#2ecc71", "#f39c12", "#e74c3c"],
        edgecolor="#ffffff",
        ax=ax
    )
    ax.set_title("Credibility Breakdown of Top 8 Most Fact-Checked Speakers (PolitiFact)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Percentage of Statements (%)", fontsize=11)
    ax.set_ylabel("Speaker", fontsize=11)
    ax.legend(title="Credibility Category", frameon=True, loc="upper right")
    plt.tight_layout()
    plot5_path = os.path.join(VIS_DIR, "speaker_credibility_analysis.png")
    plt.savefig(plot5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot5_path}")

    # ----------------------------------------------------
    # Summary JSON for Streamlit GUI & Documentation
    # ----------------------------------------------------
    summary = {
        "liar_total_claims": int(len(df_liar)),
        "liar_6class_counts": {k: int(v) for k, v in df_liar["label"].value_counts().items()},
        "liar_3class_counts": {k: int(v) for k, v in df_liar["credibility_3class"].value_counts().items()},
        "liar_mean_word_count": round(float(df_liar["word_count"].mean()), 2),
        "liar_median_word_count": int(df_liar["word_count"].median()),
        "news_total_articles": int(len(df_news)),
        "news_label_counts": {k: int(v) for k, v in df_news["label"].value_counts().items()},
        "news_mean_word_count": round(float(df_news["word_count"].mean()), 2),
        "news_median_word_count": int(df_news["word_count"].median()),
        "top_credible_bigrams": [w for w, _ in top_cred_bi],
        "top_questionable_bigrams": [w for w, _ in top_quest_bi],
    }

    summary_path = os.path.join(VIS_DIR, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved EDA Summary JSON: {summary_path}")

    print("\nExploratory Data Analysis completed successfully!")


if __name__ == "__main__":
    generate_eda()
