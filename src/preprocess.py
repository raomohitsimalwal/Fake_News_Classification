"""
Text Preprocessing and Data Cleaning Module for Fake News and Claim Classification.

Implements:
1. Missing value handling and deduplication.
2. Contraction expansion and noise removal (URLs, HTML, symbols).
3. Tokenization and lowercasing.
4. Stopword removal (preserving critical negation tokens).
5. Lemmatization (NLTK WordNetLemmatizer) & Stemming comparison.
6. Processing and saving clean datasets.
"""

import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure NLTK resources
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("wordnet")

# Initialize tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Customized stop words: preserve negation words essential for claim credibility
base_stopwords = set(stopwords.words("english"))
negation_words = {"not", "no", "nor", "neither", "never", "cannot", "without", "against", "none"}
STOP_WORDS = base_stopwords - negation_words

# Contractions dictionary for normalization
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "there's": "there is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'re": " are"
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text_lower)
    return text_lower


def clean_text_basic(text: str) -> str:
    """Clean URLs, HTML tags, special characters, and excessive whitespaces."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Expand contractions
    text = expand_contractions(text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 5. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 6. Remove non-alphabetical characters (retain letters and single spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 7. Collapse repeated whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text: str, use_lemmatization: bool = True) -> str:
    """
    Full text preprocessing pipeline:
    Cleaning -> Tokenization -> Stopword removal -> Lemmatization / Stemming.
    """
    cleaned = clean_text_basic(text)
    if not cleaned:
        return ""

    tokens = cleaned.split()

    # Remove stopwords (preserving negations)
    filtered_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Lemmatize or Stem
    if use_lemmatization:
        processed_tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    else:
        processed_tokens = [stemmer.stem(t) for t in filtered_tokens]

    return " ".join(processed_tokens)


# Map 6-class LIAR ratings into 3-tier and binary credibility for flexible analysis
LIAR_3CLASS_MAP = {
    "true": "Credible",
    "mostly-true": "Credible",
    "half-true": "Half-True",
    "barely-true": "Questionable",
    "false": "Questionable",
    "pants-fire": "Questionable"
}

LIAR_BINARY_MAP = {
    "true": "Credible",
    "mostly-true": "Credible",
    "half-true": "Questionable",
    "barely-true": "Questionable",
    "false": "Questionable",
    "pants-fire": "Questionable"
}


def clean_liar_dataset(raw_dir: str, processed_dir: str) -> pd.DataFrame:
    """Load, clean, preprocess, and combine the LIAR claim dataset splits."""
    cols = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party_affiliation", "barely_true_counts", "false_counts",
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
    ]

    splits = []
    for split_name, file_name in [("train", "liar_train.tsv"), ("test", "liar_test.tsv"), ("valid", "liar_valid.tsv")]:
        file_path = os.path.join(raw_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep="\t", header=None, names=cols)
            df["split"] = split_name
            splits.append(df)

    if not splits:
        raise FileNotFoundError("LIAR data files not found in raw_dir.")

    combined_df = pd.concat(splits, ignore_index=True)
    initial_len = len(combined_df)

    # 1. Handle missing values
    combined_df["statement"] = combined_df["statement"].fillna("")
    combined_df["speaker"] = combined_df["speaker"].fillna("unknown")
    combined_df["subject"] = combined_df["subject"].fillna("general")
    combined_df["context"] = combined_df["context"].fillna("general")

    # Filter out empty statements
    combined_df = combined_df[combined_df["statement"].str.strip() != ""]

    # 2. Remove exact duplicate statements within same label
    combined_df = combined_df.drop_duplicates(subset=["statement", "label"])

    # 3. Add mapped credibility columns
    combined_df["credibility_3class"] = combined_df["label"].map(LIAR_3CLASS_MAP)
    combined_df["credibility_binary"] = combined_df["label"].map(LIAR_BINARY_MAP)

    # 4. Text preprocessing
    print("Preprocessing LIAR claim text (lemmatization and negation-aware stopwords)...")
    combined_df["clean_statement"] = combined_df["statement"].apply(preprocess_text)

    # Create composite context feature (statement + speaker + subject)
    combined_df["rich_text"] = (
        combined_df["statement"] + " [Speaker: " + combined_df["speaker"] + "] [Topic: " + combined_df["subject"] + "]"
    )
    combined_df["clean_rich_text"] = combined_df["rich_text"].apply(preprocess_text)

    # 5. Length features for EDA
    combined_df["char_count"] = combined_df["statement"].apply(len)
    combined_df["word_count"] = combined_df["statement"].apply(lambda s: len(s.split()))

    # Filter out rows where clean_statement became empty
    combined_df = combined_df[combined_df["clean_statement"].str.strip() != ""]

    os.makedirs(processed_dir, exist_ok=True)
    out_csv = os.path.join(processed_dir, "liar_processed.csv")
    combined_df.to_csv(out_csv, index=False)

    print(f"LIAR dataset cleaned: {initial_len} -> {len(combined_df)} records saved to {out_csv}")
    return combined_df


def clean_news_dataset(raw_dir: str, processed_dir: str) -> pd.DataFrame:
    """Load, clean, and preprocess the full news articles dataset."""
    news_file = os.path.join(raw_dir, "fake_or_real_news.csv")
    if not os.path.exists(news_file):
        raise FileNotFoundError(f"News dataset {news_file} not found.")

    df = pd.read_csv(news_file)
    initial_len = len(df)

    # Handle unnamed column or ID column
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})

    # 1. Handle missing values
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Combine title and text
    df["full_text"] = df["title"] + " " + df["text"]
    df = df[df["full_text"].str.strip() != ""]

    # 2. Remove duplicates
    df = df.drop_duplicates(subset=["title", "text"])

    # 3. Preprocess text
    print("Preprocessing news articles (title + text)...")
    df["clean_title"] = df["title"].apply(preprocess_text)
    # Preprocess text (take first 3000 chars for efficiency if extremely long, or full text)
    df["clean_text"] = df["text"].apply(lambda t: preprocess_text(t[:4000]))
    df["clean_full_text"] = (df["clean_title"] + " " + df["clean_text"]).str.strip()

    # 4. Length features
    df["char_count"] = df["full_text"].apply(len)
    df["word_count"] = df["full_text"].apply(lambda s: len(s.split()))

    # Filter out empty clean text
    df = df[df["clean_full_text"].str.strip() != ""]

    os.makedirs(processed_dir, exist_ok=True)
    out_csv = os.path.join(processed_dir, "news_processed.csv")
    df.to_csv(out_csv, index=False)

    print(f"News dataset cleaned: {initial_len} -> {len(df)} records saved to {out_csv}")
    return df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

    print("=" * 60)
    print("Starting Data Cleaning & Text Preprocessing")
    print("=" * 60)

    clean_liar_dataset(RAW_DIR, PROC_DIR)
    clean_news_dataset(RAW_DIR, PROC_DIR)

    print("\nData preprocessing complete!")
