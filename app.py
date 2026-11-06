"""
TruthLens NLP / VeriClaim AI: Fake News and Claim Credibility Classification System
Main Streamlit Application for NLP Capstone Project.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Setup system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explainability import analyze_claim, analyze_news
from src.preprocess import clean_text_basic

# Page configuration
st.set_page_config(
    page_title="TruthLens NLP | Fake News & Claim Classification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Professional College Capstone Presentation UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    
    .main-header p {
        color: #e0e6ed !important;
        font-size: 1.05rem;
        margin: 0;
    }

    .disclaimer-box {
        background-color: #f8f9fa;
        border-left: 5px solid #ff9800;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 0.92rem;
        color: #555;
        margin-bottom: 20px;
    }
    
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 20px;
        border: 1px solid #e1e4e8;
    }
    
    .card-credible {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
    }
    
    .card-questionable {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
    }
    
    .card-half-true {
        background-color: #fffbeb;
        border-left: 6px solid #f59e0b;
    }

    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        color: white;
    }
    .badge-credible { background-color: #22c55e; }
    .badge-questionable { background-color: #ef4444; }
    .badge-half-true { background-color: #f59e0b; }
    .badge-real { background-color: #3b82f6; }
    .badge-fake { background-color: #ef4444; }

    .token-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .token-chip-credible { background-color: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .token-chip-questionable { background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .token-chip-neutral { background-color: #f3f4f6; color: #374151; border: 1px solid #d1d5db; }
</style>
""", unsafe_allow_html=True)


# Load summary and metrics once
@st.cache_data
def load_cached_metadata():
    metrics_file = os.path.join(BASE_DIR, "models", "evaluation_metrics.json")
    eda_file = os.path.join(BASE_DIR, "visualizations", "eda_summary.json")

    metrics, eda = {}, {}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if os.path.exists(eda_file):
        with open(eda_file, "r", encoding="utf-8") as f:
            eda = json.load(f)
    return metrics, eda


metrics_data, eda_data = load_cached_metadata()

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ TruthLens NLP: Fake News & Claim Credibility Classifier</h1>
    <p>A Full-Stack NLP Capstone System utilizing Machine Learning & TF-IDF Representations on Authentic Fact-Checking Benchmarks</p>
</div>
""", unsafe_allow_html=True)

# Mandatory Academic Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Academic & Ethical Transparency Notice:</strong> This software is a machine learning research demonstrator developed for an NLP Capstone Project. 
    It evaluates textual inputs against statistical patterns derived from historical labelled datasets (PolitiFact LIAR & McIntire Fake News Benchmarks). 
    <strong>Predictions represent probabilistic pattern matches and do not constitute absolute real-world truth or editorial determinations.</strong>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/news.png", width=120)
    st.title("Project Controls")
    st.markdown("**Project Title:** Fake News or Claim Classification")
    st.markdown("**Domain:** Natural Language Processing (NLP)")
    st.markdown("**Architecture:** TF-IDF + Calibrated Linear Models")
    
    st.divider()
    st.subheader("Benchmark Accuracy")
    if metrics_data:
        news_acc = metrics_data.get("best_news_accuracy", 0.9247) * 100
        st.metric("News Classifier Accuracy", f"{news_acc:.2f}%", help="Tested on 1,262 held-out news articles")
        claim_f1 = metrics_data.get("best_claim_macro_f1", 0.4265)
        st.metric("Claims Classifier Macro F1", f"{claim_f1:.4f}", help="Tested on 2,555 held-out fact-checking claims")
    else:
        st.info("Models ready on disk.")

    st.divider()
    st.caption("NLP Capstone Final Submission | 2026")


# Tabs for Application Sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Claim Verifier (LIAR)",
    "📰 News Article Classifier",
    "📊 Model Benchmark & Metrics",
    "📈 Dataset & EDA Explorer",
    "ℹ️ Methodology & Architecture"
])

# =============================================================================
# TAB 1: CLAIM VERIFIER (LIAR DATASET BENCHMARK)
# =============================================================================
with tab1:
    st.subheader("Fact-Checking Claim Credibility Analyzer")
    st.write(
        "Enter a political, economic, or social statement to classify its credibility based on the "
        "**PolitiFact LIAR Benchmark Dataset** (12,770+ expert-annotated claims)."
    )

    # Sample Buttons
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    if "claim_input_text" not in st.session_state:
        st.session_state.claim_input_text = ""
    if "speaker_input" not in st.session_state:
        st.session_state.speaker_input = ""
    if "topic_input" not in st.session_state:
        st.session_state.topic_input = ""

    if sample_col1.button("🟢 Sample Credible Claim"):
        st.session_state.claim_input_text = "The unemployment rate in the United States fell to 3.7 percent in September, the lowest since 1969."
        st.session_state.speaker_input = "Department of Labor"
        st.session_state.topic_input = "economy,jobs"

    if sample_col2.button("🟡 Sample Half-True Claim"):
        st.session_state.claim_input_text = "Under the new tax law, middle class families will receive a tax cut, but corporations received a permanent reduction."
        st.session_state.speaker_input = "Congressional Committee"
        st.session_state.topic_input = "taxes,economy"

    if sample_col3.button("🔴 Sample Questionable / False Claim"):
        st.session_state.claim_input_text = "The government has approved an undercover program to secretly install 5G microchips into all public municipal drinking water."
        st.session_state.speaker_input = "Anonymous Blog"
        st.session_state.topic_input = "health,conspiracy"

    if sample_col4.button("🧹 Clear Input"):
        st.session_state.claim_input_text = ""
        st.session_state.speaker_input = ""
        st.session_state.topic_input = ""

    claim_text = st.text_area(
        "Enter Statement / Claim to Analyze:",
        value=st.session_state.claim_input_text,
        height=110,
        placeholder="Type or paste a claim here (e.g. 'Presidential executive orders have increased by 40 percent this year...')"
    )

    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        speaker_val = st.text_input("Speaker / Source (Optional):", value=st.session_state.speaker_input, placeholder="e.g. White House, PolitiFact, Social Media")
    with meta_col2:
        topic_val = st.text_input("Subject / Topic (Optional):", value=st.session_state.topic_input, placeholder="e.g. economy, healthcare, foreign-policy")

    analyze_button = st.button("⚡ Analyze Claim Credibility", type="primary", use_container_width=True)

    if analyze_button:
        if not claim_text or not claim_text.strip():
            st.warning("⚠️ Please enter a claim or statement before clicking 'Analyze Claim'.")
        elif len(claim_text.strip().split()) < 3:
            st.warning("⚠️ Statement is too short. Please provide at least 3-4 words for meaningful NLP analysis.")
        else:
            with st.spinner("Processing text tokens, extracting TF-IDF features, and computing model probabilities..."):
                try:
                    result = analyze_claim(claim_text, speaker=speaker_val, topic=topic_val)
                    cat = result["predicted_category"]
                    conf = result["confidence_percentage"]
                    probs = result["probabilities"]

                    card_style = "card-credible" if cat == "Credible" else ("card-half-true" if cat == "Half-True" else "card-questionable")
                    badge_style = "badge-credible" if cat == "Credible" else ("badge-half-true" if cat == "Half-True" else "badge-questionable")
                    icon = "✅" if cat == "Credible" else ("⚖️" if cat == "Half-True" else "🚨")

                    st.markdown(f"""
                    <div class="result-card {card_style}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span class="badge {badge_style}">{icon} Predicted Category: {cat.upper()}</span>
                            <span style="font-size: 1.15rem; font-weight: 700; color: #333;">Confidence: {conf}%</span>
                        </div>
                        <p style="font-size: 1rem; color: #333; line-height: 1.5;"><strong>Model Explanation:</strong> {result['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    res_col1, res_col2 = st.columns([1.2, 1])

                    with res_col1:
                        st.subheader("Credibility Probability Breakdown")
                        prob_df = pd.DataFrame({
                            "Credibility Category": list(probs.keys()),
                            "Probability (%)": [round(p * 100, 2) for p in probs.values()]
                        }).set_index("Credibility Category")
                        st.bar_chart(prob_df, height=220)

                    with res_col2:
                        st.subheader("Influential Text Features")
                        st.caption("Tokens in your statement with strongest predictive weight:")
                        if result["salient_keywords"]:
                            chips_html = ""
                            for token, weight, lean in result["salient_keywords"]:
                                chip_class = "token-chip-credible" if lean == "Credible" else ("token-chip-questionable" if lean == "Questionable" else "token-chip-neutral")
                                chips_html += f'<span class="token-chip {chip_class}">{token} (weight: {weight:+.2f})</span> '
                            st.markdown(chips_html, unsafe_allow_html=True)
                        else:
                            st.write("No dominant domain keywords detected in dictionary.")

                        st.markdown("**Stylometric Properties:**")
                        style_m = result["stylometrics"]
                        st.write(f"- Word Count: `{style_m['word_count']}` words")
                        st.write(f"- Sensationalist ALL-CAPS: `{'Detected' if style_m['has_all_caps'] else 'None'}`")
                        st.write(f"- Exclamations: `{'Present' if style_m['has_exclamation'] else 'None'}`")
                        st.write(f"- Numerical Specificity: `{'Contains Numerals/Data' if style_m['has_numbers'] else 'No Numbers'}`")

                except Exception as e:
                    st.error(f"Error during analysis: {e}")


# =============================================================================
# TAB 2: NEWS ARTICLE CLASSIFIER (REAL VS FAKE)
# =============================================================================
with tab2:
    st.subheader("Full News Article Verifier")
    st.write(
        "Analyze full-length news articles or journalistic reports to detect deceptive styling and classify as "
        "**REAL** (Credible Journalism) or **FAKE** (Deceptive / Misinformation)."
    )

    if "news_title_input" not in st.session_state:
        st.session_state.news_title_input = ""
    if "news_text_input" not in st.session_state:
        st.session_state.news_text_input = ""

    n_sample_col1, n_sample_col2, n_sample_col3 = st.columns(3)

    if n_sample_col1.button("🟢 Load Sample Real News (Reuters)"):
        st.session_state.news_title_input = "U.S. Federal Reserve Holds Interest Rates Steady Amid Steady Economic Growth"
        st.session_state.news_text_input = (
            "WASHINGTON (Reuters) - The Federal Reserve kept interest rates unchanged on Wednesday and characterized "
            "the U.S. economy as expanding at a strong pace with the job market continuing to strengthen. "
            "In its policy statement following a two-day meeting, the central bank said economic activity has been rising "
            "at a strong rate and job gains have been strong, on average, in recent months."
        )

    if n_sample_col2.button("🔴 Load Sample Fake News Article"):
        st.session_state.news_title_input = "BOMBSHELL: Proof Emerges That Secret Cabal Operates Underneath Capital City"
        st.session_state.news_text_input = (
            "In a shocking revelation that the mainstream media is completely refusing to broadcast, whistleblowers have "
            "leaked top secret documents proving an underground network exists right beneath Washington. "
            "Insiders confirm thousands of corrupt elites gather weekly for rituals. Share this everywhere before "
            "the government wipes it from the internet forever!"
        )

    if n_sample_col3.button("🧹 Clear News Input"):
        st.session_state.news_title_input = ""
        st.session_state.news_text_input = ""

    article_title = st.text_input("News Article Headline / Title:", value=st.session_state.news_title_input, placeholder="e.g. Breaking: Economic growth exceeds forecasts in fourth quarter...")
    article_body = st.text_area("Full Article Text:", value=st.session_state.news_text_input, height=180, placeholder="Paste the news article body text here...")

    verify_news_btn = st.button("⚡ Verify News Article", type="primary", use_container_width=True)

    if verify_news_btn:
        combined_text = (article_title + " " + article_body).strip()
        if not combined_text:
            st.warning("⚠️ Please provide an article title or body text.")
        elif len(combined_text.split()) < 10:
            st.warning("⚠️ Article text is too brief for article-level classification. Please provide at least 10 words.")
        else:
            with st.spinner("Analyzing document structure, lexical density, and journalistic veracity signals..."):
                try:
                    res_news = analyze_news(article_title, article_body)
                    label = res_news["predicted_label"]
                    conf = res_news["confidence_percentage"]
                    probs = res_news["probabilities"]

                    card_style = "card-credible" if label == "REAL" else "card-questionable"
                    badge_style = "badge-real" if label == "REAL" else "badge-fake"
                    icon = "📰" if label == "REAL" else "🚫"

                    st.markdown(f"""
                    <div class="result-card {card_style}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span class="badge {badge_style}">{icon} Article Classification: {label}</span>
                            <span style="font-size: 1.15rem; font-weight: 700; color: #333;">Confidence: {conf}%</span>
                        </div>
                        <p style="font-size: 1rem; color: #333; line-height: 1.5;">{res_news['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Veracity Probabilities")
                        st.bar_chart(pd.DataFrame({
                            "Probability (%)": [round(p * 100, 2) for p in probs.values()]
                        }, index=list(probs.keys())), height=200)

                    with c2:
                        st.subheader("Stylometric & Readability Profile")
                        st_dict = res_news["stylometrics"]
                        st.write(f"- **Word Count:** `{st_dict['word_count']}` words")
                        st.write(f"- **Lexical Diversity (TTR):** `{st_dict['lexical_diversity']}` (ratio of unique to total words)")
                        st.write(f"- **Estimated Reading Time:** `{st_dict['estimated_read_time_minutes']}` minutes")

                except Exception as e:
                    st.error(f"Error during news analysis: {e}")


# =============================================================================
# TAB 3: MODEL BENCHMARK & METRICS
# =============================================================================
with tab3:
    st.subheader("Machine Learning Algorithms Benchmark & Evaluation")
    st.write(
        "In accordance with faculty capstone requirements, multiple machine learning classification algorithms "
        "were trained and evaluated across **Accuracy, Precision, Recall, and F1-score** on unseen test splits."
    )

    if metrics_data:
        st.markdown("### 1. News Classification Model Benchmark (Real vs Fake)")
        news_comp = pd.DataFrame(metrics_data["news_models_comparison"]).T
        news_comp.columns = [
            "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
            "Precision (Wt)", "Recall (Wt)", "F1-Score (Wt)"
        ]
        st.dataframe(news_comp.style.highlight_max(subset=["Accuracy", "F1-Score (Macro)"], color="#dcfce7"), use_container_width=True)

        st.markdown("### 2. Claim Credibility Benchmark (LIAR 3-Tier Multi-Class)")
        claim_comp = pd.DataFrame(metrics_data["claim_models_comparison"]).T
        claim_comp.columns = [
            "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
            "Precision (Wt)", "Recall (Wt)", "F1-Score (Wt)"
        ]
        st.dataframe(claim_comp.style.highlight_max(subset=["F1-Score (Macro)", "Accuracy"], color="#fef9c3"), use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrices (Unseen Test Sets)")
    cm_col1, cm_col2 = st.columns(2)

    cm_news_path = os.path.join(BASE_DIR, "visualizations", "confusion_matrix_news.png")
    cm_claim_path = os.path.join(BASE_DIR, "visualizations", "confusion_matrix_claims.png")
    comp_chart_path = os.path.join(BASE_DIR, "visualizations", "model_comparison_chart.png")

    with cm_col1:
        if os.path.exists(cm_news_path):
            st.image(cm_news_path, caption="Figure 1: Confusion Matrix for News Articles (Passive Aggressive Calibrated - 92.47% Accuracy)", use_container_width=True)

    with cm_col2:
        if os.path.exists(cm_claim_path):
            st.image(cm_claim_path, caption="Figure 2: Confusion Matrix for Fact-Check Claims (Logistic Regression Multi-class)", use_container_width=True)

    if os.path.exists(comp_chart_path):
        st.image(comp_chart_path, caption="Figure 3: Comparative Performance across Classification Algorithms", use_container_width=True)


# =============================================================================
# TAB 4: DATASET & EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
with tab4:
    st.subheader("Exploratory Data Analysis & Linguistic Visualizations")
    st.write(
        "Deep exploratory data analysis was conducted on both datasets to identify class distributions, "
        "text length distributions, top n-grams, word clouds, and speaker credibility tendencies."
    )

    if eda_data:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LIAR Fact-Check Claims", f"{eda_data.get('liar_total_claims', 12771):,}")
        m2.metric("Mean Claim Length", f"{eda_data.get('liar_mean_word_count', 18.05)} words")
        m3.metric("News Articles Dataset", f"{eda_data.get('news_total_articles', 6306):,}")
        m4.metric("Mean News Article Length", f"{eda_data.get('news_mean_word_count', 786.8)} words")

    st.divider()
    eda_col1, eda_col2 = st.columns(2)

    class_dist_path = os.path.join(BASE_DIR, "visualizations", "class_distribution.png")
    text_len_path = os.path.join(BASE_DIR, "visualizations", "text_length_distribution.png")
    ngrams_path = os.path.join(BASE_DIR, "visualizations", "top_ngrams_comparison.png")
    wc_path = os.path.join(BASE_DIR, "visualizations", "wordcloud_credible_vs_fake.png")
    speaker_path = os.path.join(BASE_DIR, "visualizations", "speaker_credibility_analysis.png")

    with eda_col1:
        if os.path.exists(class_dist_path):
            st.image(class_dist_path, caption="Figure 4: Credibility Class Distributions across Datasets", use_container_width=True)
        if os.path.exists(ngrams_path):
            st.image(ngrams_path, caption="Figure 6: Top Bigrams Differentiating Credible vs Questionable Claims", use_container_width=True)

    with eda_col2:
        if os.path.exists(text_len_path):
            st.image(text_len_path, caption="Figure 5: Text Length and Word Count Distributions", use_container_width=True)
        if os.path.exists(wc_path):
            st.image(wc_path, caption="Figure 7: Lexical Word Clouds for Credible vs Questionable Discourse", use_container_width=True)

    if os.path.exists(speaker_path):
        st.image(speaker_path, caption="Figure 8: Credibility Breakdown of Most Fact-Checked Speakers (PolitiFact)", use_container_width=True)


# =============================================================================
# TAB 5: METHODOLOGY & ARCHITECTURE
# =============================================================================
with tab5:
    st.subheader("NLP Capstone Project: Technical Architecture & Development Lifecycle")
    st.markdown("""
    ### 1. Problem Identification & Motivation
    Misinformation spreads rapidly across digital networks, distorting public discourse, health guidelines, and economic sentiment. 
    The objective of this project is to build an automated, interpretable NLP classification engine that analyzes textual statements 
    and predicts their credibility tier based on established fact-checking corpora.

    ### 2. End-to-End NLP Lifecycle Stages
    1. **Data Acquisition:** Programmatic ingestion of the **PolitiFact LIAR Benchmark** (12,771 claims) and **McIntire Fake News Dataset** (6,306 articles).
    2. **Data Cleaning:** Imputation of missing metadata, removal of duplicate entries, URL/HTML tag stripping, and contraction expansion.
    3. **Text Preprocessing:** Tokenization, lowercase normalization, **negation-aware stopword removal** (preserving tokens like `not`, `no`, `never` which govern truth values), and **WordNet lemmatization**.
    4. **Exploratory Data Analysis (EDA):** Statistical characterization of word frequencies, n-grams, text length distributions, and speaker credibility ratios.
    5. **Feature Engineering:** Sublinear TF-IDF (Term Frequency - Inverse Document Frequency) with unigram + bigram representations (`ngram_range=(1,2)`).
    6. **Model Development:** Multi-model comparative benchmarking across Naive Bayes, Logistic Regression, Linear SVM, Calibrated Passive Aggressive, and Random Forests.
    7. **Evaluation:** Stratified 80/20 train-test evaluation via Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
    8. **Explainability & GUI:** Streamlit interactive interface with probability breakdowns, keyword contributions, and stylometric cues.

    ### 3. Why These Models Were Selected
    - **News Classification:** The **Calibrated Passive Aggressive Classifier** achieved **92.47% accuracy**. Passive Aggressive algorithms are mathematically optimized for large-scale, high-dimensional text classification, updating weights aggressively on misclassifications while maintaining stability on correct classifications.
    - **Claim Credibility Classification:** The **Logistic Regression Classifier** with balanced class weights achieved the highest Macro F1 score (**0.4265**) on the notoriously difficult multi-class LIAR benchmark, providing calibrated odds ratios for lexical explainability.

    ### 4. Real-World Limitations & Ethical Standards
    - **Model Predictions ≠ Absolute Ground Truth:** The model predicts whether an input stylistically and contextually matches claims labelled as credible or questionable in historical training data.
    - **Temporal Drift:** New political figures or evolving events may introduce unseen vocabulary not present in the historical training split.
    - **Verification Requirement:** The tool is designed to assist human fact-checkers and media consumers, not replace rigorous journalistic verification.
    """)

# Footer
st.divider()
st.caption("TruthLens NLP Capstone System | Built with Python, Scikit-Learn, NLTK & Streamlit | 2026")
