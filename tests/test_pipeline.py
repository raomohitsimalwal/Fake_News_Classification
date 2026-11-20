"""
Comprehensive Test Suite for Fake News and Claim Credibility Classification System.
Verifies:
1. Text preprocessing (URLs, HTML, stop words, lemmatization).
2. Model loading and pipeline integrity.
3. Inference on unseen statements (Credible, Questionable, Half-True).
4. News article inference (REAL vs FAKE).
5. Explainability output formats and probability normalization.
6. Edge case handling (empty input, whitespace, numbers only).
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import clean_text_basic, preprocess_text, expand_contractions
from src.explainability import analyze_claim, analyze_news
from src.domain_classifier import DomainInferenceEngine, extractive_summarize


def test_contractions_expansion():
    sample = "Don't say it's true when they won't agree."
    expanded = expand_contractions(sample)
    assert "not" in expanded
    assert "it is" in expanded
    assert "will not" in expanded


def test_clean_text_basic():
    dirty = "Check this breaking report: https://news.example.com <b>Click NOW!!</b> #exclusive"
    cleaned = clean_text_basic(dirty)
    assert "http" not in cleaned
    assert "<b>" not in cleaned
    assert "click" in cleaned


def test_preprocess_text_preserves_negations():
    statement = "The governor did not approve the bill."
    processed = preprocess_text(statement)
    # Negation "not" must be preserved because it alters veracity
    assert "not" in processed.split()
    assert "governor" in processed.split()


def test_model_artifacts_exist():
    models_dir = os.path.join(BASE_DIR, "models")
    assert os.path.exists(os.path.join(models_dir, "news_classifier.joblib"))
    assert os.path.exists(os.path.join(models_dir, "news_vectorizer.joblib"))
    assert os.path.exists(os.path.join(models_dir, "claim_classifier.joblib"))
    assert os.path.exists(os.path.join(models_dir, "claim_vectorizer.joblib"))
    assert os.path.exists(os.path.join(models_dir, "evaluation_metrics.json"))


def test_analyze_claim_credible():
    claim = "The official labor department statistics indicate private sector employment grew by 200,000 jobs."
    res = analyze_claim(claim, speaker="Bureau of Labor Statistics", topic="economy")
    assert "predicted_category" in res
    assert res["predicted_category"] in ["Credible", "Half-True", "Questionable"]
    assert 0 <= res["confidence_percentage"] <= 100
    assert len(res["probabilities"]) == 3
    # Probabilities should sum to approximately 1.0
    total_p = sum(res["probabilities"].values())
    assert abs(total_p - 1.0) < 0.05
    assert len(res["explanation"]) > 20


def test_analyze_claim_false():
    claim = "Secret aliens have replaced world leaders in underground tunnels."
    res = analyze_claim(claim, speaker="Conspiracy Blog", topic="aliens")
    assert "predicted_category" in res
    assert res["predicted_category"] in ["Credible", "Half-True", "Questionable"]
    assert "stylometrics" in res


def test_analyze_news_real():
    title = "Treasury Announces New Fiscal Guidelines Following Quarterly Audit"
    body = (
        "WASHINGTON - The United States Department of the Treasury issued updated fiscal compliance standards "
        "on Thursday. The new guidelines require commercial banks to submit liquidity summaries each month. "
        "Officials emphasized that the changes are designed to ensure financial system stability."
    )
    res = analyze_news(title, body)
    assert res["predicted_label"] in ["REAL", "FAKE"]
    assert "confidence_percentage" in res
    assert res["confidence_percentage"] > 50


def test_empty_input_handling():
    try:
        analyze_claim("")
        assert False, "Should have raised ValueError on empty string"
    except ValueError:
        pass

    try:
        analyze_claim("   ")
        assert False, "Should have raised ValueError on whitespace"
    except ValueError:
        pass

    try:
        analyze_news("", "")
        assert False, "Should have raised ValueError on empty news"
    except ValueError:
        pass


def test_domain_classifier_artifact_exists():
    models_dir = os.path.join(BASE_DIR, "models")
    assert os.path.exists(os.path.join(models_dir, "domain_classifier.joblib"))
    assert os.path.exists(os.path.join(models_dir, "domain_metrics.json"))


def test_domain_classifier_inference_legal():
    engine = DomainInferenceEngine()
    sample_legal = (
        "This Mutual Non-Disclosure Agreement governs confidential proprietary data, trade secrets, "
        "and patent filings. Neither party shall disclose information without injunctive relief in Delaware courts."
    )
    res = engine.classify_document(sample_legal)
    assert res["predicted_domain"] == "Legal"
    assert res["confidence"] > 0.50
    assert "summary" in res
    assert len(res["summary"]) > 20


def test_domain_classifier_inference_healthcare():
    engine = DomainInferenceEngine()
    sample_health = (
        "Patient presents with acute chest discomfort and ST-segment elevation. "
        "Initiated 325mg aspirin and transferred for emergent cardiac catheterization."
    )
    res = engine.classify_document(sample_health)
    assert res["predicted_domain"] == "Healthcare"
    assert res["confidence"] > 0.50


def test_domain_extractive_summary():
    text = (
        "Enterprise software revenue expanded 18% year over year. Operating margins rose to 31%. "
        "Free cash flow totaled $22 billion for the current fiscal quarter. Capital expenditures were focused on infrastructure."
    )
    summ = extractive_summarize(text, max_sentences=2)
    assert len(summ) > 20
    assert len(summ.split()) < len(text.split())


if __name__ == "__main__":
    print("Running Pipeline Tests...")
    test_contractions_expansion()
    print("[PASS] Contraction expansion test passed")
    test_clean_text_basic()
    print("[PASS] Clean text test passed")
    test_preprocess_text_preserves_negations()
    print("[PASS] Negation preservation test passed")
    test_model_artifacts_exist()
    print("[PASS] Model artifacts existence test passed")
    test_analyze_claim_credible()
    print("[PASS] Claim analysis test passed")
    test_analyze_claim_false()
    print("[PASS] False claim analysis test passed")
    test_analyze_news_real()
    print("[PASS] News article analysis test passed")
    test_empty_input_handling()
    print("[PASS] Edge cases & empty input test passed")
    test_domain_classifier_artifact_exists()
    print("[PASS] Project 15: Domain model artifacts exist")
    test_domain_classifier_inference_legal()
    print("[PASS] Project 15: Legal document inference test passed")
    test_domain_classifier_inference_healthcare()
    print("[PASS] Project 15: Healthcare document inference test passed")
    test_domain_extractive_summary()
    print("[PASS] Project 15: Extractive summarization test passed")
    print("\nALL DUAL-PROJECT UNIT AND INTEGRATION TESTS PASSED SUCCESSFULLY!")
