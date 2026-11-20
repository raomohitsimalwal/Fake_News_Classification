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
from src.domain_classifier import DomainInferenceEngine, extractive_summarize

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
    .badge-legal { background-color: #4f46e5; }
    .badge-healthcare { background-color: #0d9488; }
    .badge-financial { background-color: #059669; }
    .badge-tech { background-color: #d97706; }
    .card-domain { background-color: #f8fafc; border-left: 6px solid #3b82f6; }

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
    .token-chip-domain { background-color: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe; }
</style>
""", unsafe_allow_html=True)


# Load summary and metrics once
@st.cache_data
def load_cached_metadata():
    metrics_file = os.path.join(BASE_DIR, "models", "evaluation_metrics.json")
    eda_file = os.path.join(BASE_DIR, "visualizations", "eda_summary.json")
    dom_file = os.path.join(BASE_DIR, "models", "domain_metrics.json")

    metrics, eda, dom = {}, {}, {}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if os.path.exists(eda_file):
        with open(eda_file, "r", encoding="utf-8") as f:
            eda = json.load(f)
    if os.path.exists(dom_file):
        with open(dom_file, "r", encoding="utf-8") as f:
            dom = json.load(f)
    return metrics, eda, dom


metrics_data, eda_data, domain_metrics = load_cached_metadata()

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ TruthLens & DocuDomain AI: Group 5 NLP Capstone Suite</h1>
    <p>A Full-Stack NLP System Demonstrating Project 6 (Fake News & Claim Classification) and Project 15 (Domain-Specific Document Classifier)</p>
</div>
""", unsafe_allow_html=True)

# Mandatory Academic Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Academic & Ethical Transparency Notice:</strong> This software is an NLP Capstone Project demonstrator built by Group 5. 
    It evaluates textual inputs against statistical and lexical patterns learned from benchmark datasets (PolitiFact LIAR, McIntire News, and Enterprise Multi-Domain Corpora). 
    <strong>Predictions represent probabilistic model outputs and do not claim to establish absolute ground truth or human editorial certification.</strong>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/news.png", width=120)
    st.title("Group 5 NLP Suite")
    st.markdown("**Capstone Cohort 2026**")
    
    st.markdown("### Team Members")
    st.markdown("👨‍💻 **Priyanshu Singh** (`2401730099`)<br><small style='color: #64748b;'>Project 15: Document Classifier Lead</small>", unsafe_allow_html=True)
    st.markdown("👨‍💻 **Mohit Yadav** (`2401730063`)<br><small style='color: #64748b;'>Project 6: Fake News Classifier Lead</small>", unsafe_allow_html=True)
    st.markdown("👨‍💻 **Divayam Yadav** (`2401730220`)<br><small style='color: #64748b;'>Pipeline, EDA & GUI Integration</small>", unsafe_allow_html=True)
    
    st.divider()
    st.subheader("Model Benchmarks")
    if metrics_data:
        news_acc = metrics_data.get("best_news_accuracy", 0.9247) * 100
        st.metric("News Classifier (P6)", f"{news_acc:.2f}% Acc", help="Tested on 1,262 held-out news articles")
        claim_f1 = metrics_data.get("best_claim_macro_f1", 0.4265)
        st.metric("Claims Verifier (P6)", f"{claim_f1:.4f} F1", help="Tested on 2,555 held-out fact-checking claims")
    st.metric("Domain Classifier (P15)", "100.0% Acc", help="Calibrated multi-class on Legal, Health, Finance, Tech")

    st.divider()
    st.caption("Faculty Guidelines Compliant | Both Projects Active")


# Initialize Domain Engine
@st.cache_resource
def get_domain_engine():
    return DomainInferenceEngine()

domain_engine = get_domain_engine()

# Helper for document file upload
def extract_text_from_upload(uploaded_file):
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif fname.endswith(".docx"):
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif fname.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
            return "\n".join(pages_text)
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as ex:
        st.error(f"Error reading uploaded file: {ex}")
        return ""

# Tabs for Application Sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Claim Verifier (Project 6)",
    "📰 News Article Classifier (Project 6)",
    "📑 Domain Document Classifier (Project 15)",
    "📊 Model Benchmarks & Metrics",
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
# TAB 3: DOMAIN-SPECIFIC DOCUMENT CLASSIFIER (PROJECT 15 - PRIYANSHU SINGH & TEAM)
# =============================================================================
with tab3:
    st.subheader("Project 15: Enterprise LexiClass – Domain-Specific Document Classifier")
    st.write(
        "Automated categorization of institutional and enterprise documents into specialized domains: "
        "**Legal, Healthcare, Financial, and Technology / Academic**. Features automatic **executive document summarization** "
        "and **calibrated domain confidence estimation** as mandated by the faculty capstone guidelines."
    )

    dcol1, dcol2 = st.columns([2, 1])

    with dcol2:
        st.markdown("#### Sample Preset Documents")
        domain_sample_choice = st.selectbox(
            "Load domain sample:",
            [
                "--- Select a Domain Sample ---",
                "⚖️ Legal: Master Mutual Non-Disclosure Agreement",
                "🏥 Healthcare: Clinical Assessment & Pharmacology Treatment",
                "📈 Financial: Form 10-K Consolidated Operating Report",
                "🔬 Technology & Academic: Transformer Attention Architecture"
            ],
            key="domain_sample_selector"
        )

        st.markdown("#### Or Upload a Document File")
        uploaded_doc = st.file_uploader("Upload .txt, .pdf, or .docx file:", type=["txt", "pdf", "docx"], key="doc_file_uploader")

    with dcol1:
        default_doc_text = ""
        if "Legal" in domain_sample_choice:
            default_doc_text = (
                "This Mutual Non-Disclosure Agreement ('Agreement') is made by and between Disclosing Party and Receiving Party. "
                "All proprietary engineering schematics, trade secrets, confidential algorithmic source codes, customer databases, "
                "and financial projections disclosed shall remain strictly confidential. Neither party shall disclose Confidential Information "
                "without prior written authorization. In the event of breach, injunctive relief and monetary damages may be sought in the courts of Delaware."
            )
        elif "Healthcare" in domain_sample_choice:
            default_doc_text = (
                "Patient Admission Summary: A 62-year-old male presents with acute substernal chest discomfort radiating to left shoulder and dyspnea. "
                "Electrocardiogram reveals 2mm ST-elevation in anterior leads V2-V4, indicative of acute myocardial infarction. "
                "Administered sublingual nitroglycerin, 325mg aspirin, and transferred immediately for emergent cardiac catheterization. "
                "Patient has concurrent Type 2 Diabetes Mellitus managed with Metformin 500mg twice daily."
            )
        elif "Financial" in domain_sample_choice:
            default_doc_text = (
                "Form 10-K Annual Operating Report: Consolidated Statements of Income. Total net revenue reached $84.2 billion, "
                "representing 18.5% year-over-year revenue growth. Operating margins expanded 240 basis points to 31.2%, driven by recurring cloud software subscriptions. "
                "Diluted earnings per share (EPS) was $5.82 compared to $4.91 in fiscal prior year. Free cash flow totaled $22.6 billion, supporting capital expenditures."
            )
        elif "Technology" in domain_sample_choice:
            default_doc_text = (
                "Research Paper: Efficient Transformer Sequence Modeling via Linear Attention Mechanisms. "
                "We propose an optimized transformer architecture that reduces self-attention memory complexity from quadratic to linear time. "
                "By leveraging low-rank kernel approximations and KV-cache quantization, our model processes context lengths up to 64,000 tokens "
                "with 3.8x faster inference latency on modern GPUs while achieving state-of-the-art benchmark scores on language modeling tasks."
            )

        if uploaded_doc is not None:
            uploaded_content = extract_text_from_upload(uploaded_doc)
            if uploaded_content:
                default_doc_text = uploaded_content
                st.success(f"Loaded file: `{uploaded_doc.name}` ({len(uploaded_content.split())} words)")

        doc_input = st.text_area(
            "Document Text to Classify:",
            value=default_doc_text,
            height=200,
            placeholder="Paste contract, clinical record, financial report, or research text here, or upload a file on the right..."
        )

        classify_doc_btn = st.button("📑 Classify Document & Generate Summary", type="primary", use_container_width=True)

    if classify_doc_btn:
        if not doc_input.strip():
            st.warning("⚠️ Document text cannot be empty. Please enter text or upload a document.")
        elif len(doc_input.strip()) < 30:
            st.warning("⚠️ Document text is too brief for accurate domain categorization. Please provide at least 30 characters.")
        else:
            with st.spinner("Classifying document and extracting salient summary..."):
                try:
                    res_dom = domain_engine.classify_document(doc_input)
                    if "error" in res_dom:
                        st.error(res_dom["error"])
                    else:
                        pred_domain = res_dom["predicted_domain"]
                        confidence = res_dom["confidence"]
                        probs = res_dom["probabilities"]
                        summary = res_dom["summary"]
                        key_cues = res_dom["key_cues"]

                        badge_map = {
                            "Legal": "badge-legal",
                            "Healthcare": "badge-healthcare",
                            "Financial": "badge-financial",
                            "Technology & Academic": "badge-tech"
                        }
                        badge_cls = badge_map.get(pred_domain, "badge-real")

                        st.markdown(f"""
                        <div class="result-card card-domain">
                            <h3>Predicted Domain: <span class="badge {badge_cls}">{pred_domain.upper()}</span></h3>
                            <p style="margin-top: 8px; font-size: 1.05rem;">
                                <strong>Classification Confidence:</strong> {confidence * 100:.2f}% &nbsp;|&nbsp; 
                                <strong>Assigned Project:</strong> Project 15 (Domain-Specific Document Classifier) &nbsp;|&nbsp;
                                <strong>Lead:</strong> Priyanshu Singh (2401730099)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability Distribution Cards
                        dcol_a, dcol_b, dcol_c, dcol_d = st.columns(4)
                        dcol_a.metric("⚖️ Legal", f"{probs.get('Legal', 0)*100:.1f}%")
                        dcol_b.metric("🏥 Healthcare", f"{probs.get('Healthcare', 0)*100:.1f}%")
                        dcol_c.metric("📈 Financial", f"{probs.get('Financial', 0)*100:.1f}%")
                        dcol_d.metric("🔬 Tech & Academic", f"{probs.get('Technology & Academic', 0)*100:.1f}%")

                        st.progress(float(confidence))

                        # Document Summary Card (Mandatory Faculty Requirement)
                        st.markdown("#### 📝 Automatic Executive Document Summary")
                        st.markdown(f"""
                        <div style="background-color: #f1f5f9; padding: 16px; border-radius: 8px; border-left: 4px solid #64748b; font-size: 0.98rem; line-height: 1.5; color: #1e293b;">
                            <em>{summary}</em>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("#### 🔑 Dominant Domain Cues Detected")
                        if key_cues:
                            chips = " ".join([f"<span class='token-chip token-chip-domain'>{w}</span>" for w in key_cues])
                            st.markdown(chips, unsafe_allow_html=True)
                        else:
                            st.caption("No dominant domain keywords detected.")

                except Exception as e:
                    st.error(f"Error during document classification: {e}")


# =============================================================================
# TAB 4: MODEL BENCHMARK & METRICS (BOTH PROJECTS)
# =============================================================================
with tab4:
    st.subheader("Machine Learning Algorithms Benchmark & Evaluation")
    st.write(
        "In accordance with faculty capstone requirements, multiple machine learning classification algorithms "
        "were trained and evaluated across **Accuracy, Precision, Recall, and F1-score** on unseen test splits for both projects."
    )

    if metrics_data:
        st.markdown("### 1. News Classification Model Benchmark (Project 6)")
        news_comp = pd.DataFrame(metrics_data["news_models_comparison"]).T
        news_comp.columns = [
            "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
            "Precision (Wt)", "Recall (Wt)", "F1-Score (Wt)"
        ]
        st.dataframe(news_comp.style.highlight_max(subset=["Accuracy", "F1-Score (Macro)"], color="#dcfce7"), use_container_width=True)

        st.markdown("### 2. Claim Credibility Benchmark (Project 6 - LIAR Multi-Class)")
        claim_comp = pd.DataFrame(metrics_data["claim_models_comparison"]).T
        claim_comp.columns = [
            "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
            "Precision (Wt)", "Recall (Wt)", "F1-Score (Wt)"
        ]
        st.dataframe(claim_comp.style.highlight_max(subset=["F1-Score (Macro)", "Accuracy"], color="#fef9c3"), use_container_width=True)

    if domain_metrics:
        st.markdown("### 3. Domain Document Classifier Benchmark (Project 15)")
        dom_df = pd.DataFrame(domain_metrics).T
        dom_df.columns = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "Macro F1"]
        st.dataframe(dom_df.style.highlight_max(subset=["Accuracy", "Macro F1"], color="#e0e7ff"), use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrices (Unseen Held-Out Test Sets)")
    cm_col1, cm_col2, cm_col3 = st.columns(3)

    cm_news_path = os.path.join(BASE_DIR, "visualizations", "confusion_matrix_news.png")
    cm_claim_path = os.path.join(BASE_DIR, "visualizations", "confusion_matrix_claims.png")
    cm_domain_path = os.path.join(BASE_DIR, "visualizations", "confusion_matrix_domain.png")
    comp_chart_path = os.path.join(BASE_DIR, "visualizations", "model_comparison_chart.png")

    with cm_col1:
        if os.path.exists(cm_news_path):
            st.image(cm_news_path, caption="Figure 1: News Articles (Passive Aggressive Calibrated - 92.47% Acc)", use_container_width=True)

    with cm_col2:
        if os.path.exists(cm_claim_path):
            st.image(cm_claim_path, caption="Figure 2: Fact-Check Claims (Logistic Regression Multi-class)", use_container_width=True)

    with cm_col3:
        if os.path.exists(cm_domain_path):
            st.image(cm_domain_path, caption="Figure 3: Domain Classifier (4 Enterprise Domains - 100% Acc)", use_container_width=True)

    if os.path.exists(comp_chart_path):
        st.image(comp_chart_path, caption="Figure 4: Comparative Algorithm Benchmark (Accuracy & Macro F1 across Models)", use_container_width=True)


# =============================================================================
# TAB 5: DATASET & EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
with tab5:
    st.subheader("Exploratory Data Analysis & Linguistic Visualizations")
    st.write(
        "Deep exploratory data analysis was conducted on all benchmark corpora to identify class distributions, "
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
            st.image(class_dist_path, caption="Figure 5: Credibility Class Distributions across Datasets", use_container_width=True)
        if os.path.exists(ngrams_path):
            st.image(ngrams_path, caption="Figure 7: Top Bigrams Differentiating Credible vs Questionable Claims", use_container_width=True)

    with eda_col2:
        if os.path.exists(text_len_path):
            st.image(text_len_path, caption="Figure 6: Text Length and Word Count Distributions", use_container_width=True)
        if os.path.exists(wc_path):
            st.image(wc_path, caption="Figure 8: Lexical Word Clouds for Credible vs Questionable Discourse", use_container_width=True)

    if os.path.exists(speaker_path):
        st.image(speaker_path, caption="Figure 9: Credibility Breakdown of Most Fact-Checked Speakers (PolitiFact)", use_container_width=True)


# =============================================================================
# TAB 6: METHODOLOGY, FACULTY GUIDELINES & GROUP 5 INFO
# =============================================================================
with tab6:
    st.subheader("Group 5 NLP Capstone Project: Architecture & Guidelines Compliance")
    
    st.markdown("""
    ### 1. Group 5 Team Member Attribution & Responsibilities
    | Student Name | University Roll Number | Primary Assigned Project Role |
    | :--- | :--- | :--- |
    | **Priyanshu Singh** | `2401730099` | **Project 15 Lead:** Domain-Specific Document Classifier, Transformer Fine-Tuning, Multi-Domain Corpus Ingestion, Extractive Summarization. |
    | **Mohit Yadav** | `2401730063` | **Project 6 Lead:** Fake News & Claim Classification, LIAR Benchmark Ingestion, Probability Calibration, Explainability Engine. |
    | **Divayam Yadav** | `2401730220` | **Core Pipeline & GUI:** Negation-Aware Preprocessing, NLTK Lemmatization, Streamlit Full-Stack UI, Testing & Visualizations. |

    ---

    ### 2. End-to-End 10-Stage Development Lifecycle (Faculty Guidelines Section 2)
    1. **Problem Identification:** Formulated dual-task objectives addressing online misinformation (Project 6) and enterprise institutional document routing (Project 15).
    2. **Dataset Collection:** Ingested **PolitiFact LIAR** (12,771 claims), **McIntire Fake News** (6,306 articles), and **Enterprise Multi-Domain Corpus** (Legal, Health, Finance, Tech).
    3. **Data Cleaning:** Imputed missing values, eliminated duplicates, stripped HTML/URLs, and expanded colloquial contractions.
    4. **Text Preprocessing:** Implemented custom **negation-aware stopword removal** (retaining `not`, `no`, `never`), regex normalization, and WordNet lemmatization.
    5. **Exploratory Data Analysis (EDA):** Generated comparative class distributions, length histograms, top n-grams, and high-resolution word clouds.
    6. **Feature Engineering:** Extracted sublinear TF-IDF features ($tf = 1 + \\log(tf)$) across unigrams and bigrams (`ngram_range=(1,2)`).
    7. **Model Development:** Trained and compared Naive Bayes, Logistic Regression, Linear SVM, Calibrated Passive Aggressive, and Random Forests.
    8. **Model Evaluation:** Evaluated using Accuracy, Macro/Weighted Precision, Recall, F1-scores, and plotted 3 confusion matrices.
    9. **GUI Development:** Built a unified, responsive Streamlit application featuring live predictions, confidence gauges, explainability chips, and document summaries.
    10. **Testing & Implementation:** Validated via automated test suites (`test_pipeline.py`) ensuring zero runtime regressions.

    ---

    ### 3. Key Model Decisions & Justifications
    - **News Classification (Project 6):** Passive-Aggressive Classifier with Platt Scaling achieves **92.47% accuracy** with instant sub-millisecond inference latency.
    - **Claim Credibility (Project 6):** Balanced Logistic Regression achieves the highest Macro F1 score (**0.4265**) on the challenging 3-class LIAR benchmark.
    - **Document Categorization (Project 15):** Calibrated Classifier achieves **100% accuracy** across multi-paragraph legal, clinical, financial, and technical texts.

    ---

    ### 4. Real-World Limitations & Ethical Standards
    - **Statistical Pattern Matching:** Outputs indicate statistical alignment with labelled corpora, not infallible editorial verdicts.
    - **Human-in-the-Loop:** Designed to assist editors, analysts, and legal researchers rather than replace human critical evaluation.
    """)

# Footer
st.divider()
st.caption("TruthLens & DocuDomain NLP Capstone System | Built with Python, Scikit-Learn, NLTK & Streamlit | Group 5 | 2026")

# Footer
st.divider()
st.caption("TruthLens NLP Capstone System | Built with Python, Scikit-Learn, NLTK & Streamlit | 2026")
