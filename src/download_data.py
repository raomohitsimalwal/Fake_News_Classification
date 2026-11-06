"""
Data Acquisition Module for Fake News and Claim Classification.
Downloads authentic publicly available benchmark datasets:
1. LIAR Fact-Checking Dataset (William Yang Wang, ACL 2017)
2. Fake or Real News Dataset (George McIntire / Lutz Hamel)
"""

import os
import sys
import urllib.request
import pandas as pd


LIAR_BASE_URL = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/"
LIAR_FILES = {
    "liar_train.tsv": "train.tsv",
    "liar_test.tsv": "test.tsv",
    "liar_valid.tsv": "valid.tsv"
}

NEWS_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
NEWS_FILE = "fake_or_real_news.csv"

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with user-agent and progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[EXISTS] {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes)")
        return True

    print(f"[DOWNLOADING] {url} -> {dest_path}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[DOWNLOADED] {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}", file=sys.stderr)
        return False


def download_all_datasets():
    """Download all benchmark datasets for claims and news articles."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print("=" * 60)
    print("Starting Dataset Acquisition for Fake News & Claim Classification")
    print("=" * 60)

    # 1. Download LIAR Dataset splits
    print("\n--- 1. LIAR Benchmark Dataset (Fact-checking Claims) ---")
    for local_name, remote_name in LIAR_FILES.items():
        url = LIAR_BASE_URL + remote_name
        dest = os.path.join(RAW_DATA_DIR, local_name)
        success = download_file(url, dest)
        if not success:
            print(f"Warning: Could not download {remote_name}")

    # 2. Download News Articles Dataset
    print("\n--- 2. Fake or Real News Dataset (Full News Articles) ---")
    news_dest = os.path.join(RAW_DATA_DIR, NEWS_FILE)
    success = download_file(NEWS_URL, news_dest)
    if not success:
        print(f"Warning: Could not download {NEWS_FILE}")

    # Verify downloads
    print("\n--- Verification of Downloaded Datasets ---")
    liar_cols = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party_affiliation", "barely_true_counts", "false_counts",
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    train_path = os.path.join(RAW_DATA_DIR, "liar_train.tsv")
    if os.path.exists(train_path):
        df_liar = pd.read_csv(train_path, sep="\t", header=None, names=liar_cols)
        print(f"LIAR Train Dataset: {df_liar.shape[0]} claims, {df_liar.shape[1]} features.")
        print(f"LIAR Credibility Labels: {dict(df_liar['label'].value_counts())}")

    if os.path.exists(news_dest):
        df_news = pd.read_csv(news_dest)
        print(f"News Dataset: {df_news.shape[0]} articles, {df_news.shape[1]} columns ({list(df_news.columns)}).")
        print(f"News Labels: {dict(df_news['label'].value_counts())}")

    print("\nDataset acquisition completed successfully!")


if __name__ == "__main__":
    download_all_datasets()
