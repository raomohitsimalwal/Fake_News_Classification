"""
Prediction and Explainability Module for Fake News and Claim Classification.

Provides:
1. End-to-end inference on user inputs (claims or news articles).
2. Prediction probability distributions and confidence scores.
3. Feature attribution / lexical importance explanation (identifies salient keywords).
4. Stylometric / linguistic pattern analysis (clickbait cues, emotional polarity markers, lexical density).
5. Academic transparency disclaimer.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from src.preprocess import preprocess_text, clean_text_basic
except ImportError:
    from preprocess import preprocess_text, clean_text_basic
MODEL_DIR = os.path.join(BASE_DIR, "models")

_news_model = None
_news_vectorizer = None
_claim_model = None
_claim_vectorizer = None


def load_artifacts():
    """Lazy load models and vectorizers to optimize startup time."""
    global _news_model, _news_vectorizer, _claim_model, _claim_vectorizer
    if _news_model is None:
        _news_model = joblib.load(os.path.join(MODEL_DIR, "news_classifier.joblib"))
        _news_vectorizer = joblib.load(os.path.join(MODEL_DIR, "news_vectorizer.joblib"))
    if _claim_model is None:
        _claim_model = joblib.load(os.path.join(MODEL_DIR, "claim_classifier.joblib"))
        _claim_vectorizer = joblib.load(os.path.join(MODEL_DIR, "claim_vectorizer.joblib"))


def get_salient_features_claim(preprocessed_text: str, top_n: int = 5) -> List[Tuple[str, float, str]]:
    """
    Extract the most influential tokens in the claim based on Logistic Regression weights.
    Returns list of (token, weight, leaning_direction).
    """
    tokens = preprocessed_text.split()
    feature_names = _claim_vectorizer.get_feature_names_out()
    vocab = _claim_vectorizer.vocabulary_
    classes = _claim_model.classes_

    # For LogisticRegression, coef_ shape is (n_classes, n_features)
    coefs = _claim_model.coef_

    contributions = []
    for token in set(tokens):
        if token in vocab:
            idx = vocab[token]
            # Get max coefficient class
            token_coefs = coefs[:, idx]
            top_class_idx = np.argmax(token_coefs)
            leaning_class = classes[top_class_idx]
            weight = float(token_coefs[top_class_idx])
            contributions.append((token, round(weight, 3), leaning_class))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


def analyze_claim(claim_text: str, speaker: str = "", topic: str = "") -> Dict[str, Any]:
    """
    Analyze a user-provided fact-checking claim against the PolitiFact LIAR trained model.
    """
    load_artifacts()

    if not claim_text or not claim_text.strip():
        raise ValueError("Claim statement cannot be empty.")

    # 1. Text preprocessing
    cleaned_statement = preprocess_text(claim_text)
    if not cleaned_statement:
        cleaned_statement = clean_text_basic(claim_text)

    # 2. Rich composite text
    rich_text = claim_text
    if speaker or topic:
        rich_text += f" [Speaker: {speaker or 'unknown'}] [Topic: {topic or 'general'}]"
    cleaned_rich = preprocess_text(rich_text)
    if not cleaned_rich:
        cleaned_rich = cleaned_statement

    # 3. Vectorization
    X_vec = _claim_vectorizer.transform([cleaned_rich])

    # 4. Predict probabilities & category
    probs = _claim_model.predict_proba(X_vec)[0]
    classes = _claim_model.classes_

    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
    predicted_class = classes[np.argmax(probs)]
    confidence_score = round(float(np.max(probs)) * 100, 1)

    # 5. Salient Keywords
    salient_words = get_salient_features_claim(cleaned_rich, top_n=6)

    # 6. Linguistic Stylometrics
    raw_words = claim_text.split()
    word_count = len(raw_words)
    has_exclamation = "!" in claim_text
    has_all_caps = any(w.isupper() and len(w) > 2 for w in raw_words)
    has_numbers = any(char.isdigit() for char in claim_text)

    # 7. Generate explanation narrative
    if predicted_class == "Credible":
        explanation = (
            f"The model classified this claim as 'Credible' with {confidence_score}% confidence. "
            f"The claim shares linguistic framing and terminology frequently corroborated in fact-checked datasets. "
        )
    elif predicted_class == "Questionable":
        explanation = (
            f"The model flagged this claim as 'Questionable' with {confidence_score}% confidence. "
            f"The statement exhibits rhetorical characteristics and phrasing statistically aligned with false or heavily disputed statements. "
        )
    else:  # Half-True
        explanation = (
            f"The model assessed this claim as 'Half-True / Partially Credible' with {confidence_score}% confidence. "
            f"The claim contains elements of factual discussion mixed with uncorroborated, nuanced, or context-dependent assertions. "
        )

    if has_all_caps:
        explanation += " Note: Excessive capitalization was detected, which is often associated with sensationalist or emotionally charged discourse."

    return {
        "input_claim": claim_text,
        "predicted_category": predicted_class,
        "confidence_percentage": confidence_score,
        "probabilities": prob_dict,
        "salient_keywords": salient_words,
        "explanation": explanation,
        "stylometrics": {
            "word_count": word_count,
            "has_exclamation": has_exclamation,
            "has_all_caps": has_all_caps,
            "has_numbers": has_numbers
        },
        "disclaimer": (
            "Academic Disclaimer: This verdict is generated by an NLP machine learning model trained on historical "
            "fact-checking data (PolitiFact LIAR Benchmark). It reflects statistical patterns in training text and does not "
            "constitute an absolute guarantee of real-world veracity."
        )
    }


def analyze_news(title: str, text: str) -> Dict[str, Any]:
    """
    Analyze a full news article (title + body) against the News Article Classifier.
    """
    load_artifacts()

    full_raw = (title + " " + text).strip()
    if not full_raw:
        raise ValueError("News article title or text must be provided.")

    clean_content = preprocess_text(full_raw)
    if not clean_content:
        clean_content = clean_text_basic(full_raw)

    X_vec = _news_vectorizer.transform([clean_content])

    # Model supports predict_proba
    probs = _news_model.predict_proba(X_vec)[0]
    classes = _news_model.classes_

    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
    predicted_label = classes[np.argmax(probs)]
    confidence_score = round(float(np.max(probs)) * 100, 1)

    # Stylometrics
    raw_words = full_raw.split()
    word_count = len(raw_words)
    unique_words = len(set(raw_words))
    lexical_diversity = round(unique_words / max(word_count, 1), 3)
    reading_time_mins = round(word_count / 200, 1)

    if predicted_label == "REAL":
        explanation = (
            f"The article is classified as 'REAL / Credible Journalism' with {confidence_score}% confidence. "
            f"The text features formal journalistic syntax, objective reporting conventions, and balanced vocabulary."
        )
    else:
        explanation = (
            f"The article is classified as 'FAKE / Deceptive News' with {confidence_score}% confidence. "
            f"The article displays lexical markers, emotive persuasion, or narrative styling statistically associated with misinformation."
        )

    return {
        "title": title,
        "predicted_label": predicted_label,
        "confidence_percentage": confidence_score,
        "probabilities": prob_dict,
        "explanation": explanation,
        "stylometrics": {
            "word_count": word_count,
            "lexical_diversity": lexical_diversity,
            "estimated_read_time_minutes": reading_time_mins
        },
        "disclaimer": (
            "Academic Disclaimer: Predictions represent algorithmic pattern matching against benchmark datasets. "
            "Always consult primary verified journalistic sources for definitive verification."
        )
    }


if __name__ == "__main__":
    print("Testing Explainability Engine on unseen claims and news:")

    sample_credible_claim = "The official unemployment rate dropped to 3.7 percent in the third quarter of 2018."
    res1 = analyze_claim(sample_credible_claim, speaker="Bureau of Labor Statistics", topic="economy")
    print(f"\nClaim: {sample_credible_claim}")
    print(f"Prediction: {res1['predicted_category']} ({res1['confidence_percentage']}%)")
    print(f"Probabilities: {res1['probabilities']}")
    print(f"Salient Keywords: {res1['salient_keywords']}")

    sample_false_claim = "The secret government agency is putting mind-control chemicals in all tap water across the country."
    res2 = analyze_claim(sample_false_claim, speaker="Unknown Blog", topic="conspiracy")
    print(f"\nClaim: {sample_false_claim}")
    print(f"Prediction: {res2['predicted_category']} ({res2['confidence_percentage']}%)")
    print(f"Probabilities: {res2['probabilities']}")
    print(f"Salient Keywords: {res2['salient_keywords']}")
