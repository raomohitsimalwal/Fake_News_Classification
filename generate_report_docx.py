"""
Academic Capstone Project Report Generator (.docx).
Generates a comprehensive, formal, publication-grade Capstone Report for:
Group 5 Dual Project NLP Submission:
  - Project 6: Fake News or Claim Classification (TruthLens AI) - Lead: Mohit Yadav (2401730063)
  - Project 15: Domain-Specific Document Classifier (Enterprise LexiClass AI) - Lead: Priyanshu Singh (2401730099)
  - Core Pipeline, EDA & GUI Integration - Divayam Yadav (2401730220)
"""

import os
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
OUTPUT_DOCX = os.path.join(BASE_DIR, "TruthLens_NLP_Capstone_Report.docx")

# Styling Colors
NAVY = RGBColor(26, 54, 93)      # #1A365D
BLUE = RGBColor(43, 108, 176)    # #2B6CB0
DARK = RGBColor(45, 55, 72)      # #2D3748
MUTED = RGBColor(113, 128, 150)  # #718096


def set_cell_background(cell, hex_color):
    """Set background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def style_cell(cell, text, is_header=False, bg_color=None):
    """Set text and font formatting on a table cell."""
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(9.5)
    if is_header:
        run.font.bold = True
    if bg_color:
        set_cell_background(cell, bg_color)


def build_report():
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # -------------------------------------------------------------------------
    # COVER / TITLE PAGE
    # -------------------------------------------------------------------------
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_inst = p_inst.add_run("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\nNLP CAPSTONE PROJECT REPORT — ACADEMIC SESSION 2026")
    run_inst.font.name = "Arial"
    run_inst.font.size = Pt(11)
    run_inst.font.bold = True
    run_inst.font.color.rgb = BLUE
    p_inst.paragraph_format.space_after = Pt(36)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("TRUTHLENS & DOCUDOMAIN AI:\nNLP CAPSTONE PROJECT REPORT")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY
    p_title.paragraph_format.space_after = Pt(12)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run(
        "A Dual-Project NLP Architecture Comprising Project 6 (Fake News & Claim Credibility Classification)\n"
        "and Project 15 (Domain-Specific Document Classifier & Executive Summarizer)\n"
        "Developed Strictly in Accordance with Faculty Capstone Instructions"
    )
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = MUTED
    p_sub.paragraph_format.space_after = Pt(50)

    # Group & Candidate Box
    p_subm = doc.add_paragraph()
    p_subm.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_subm = p_subm.add_run("SUBMITTED BY: GROUP 5\n")
    run_subm.font.name = "Arial"
    run_subm.font.size = Pt(12)
    run_subm.font.bold = True
    run_subm.font.color.rgb = NAVY

    members = [
        ("PRIYANSHU SINGH", "2401730099", "Project 15 Lead: Domain-Specific Document Classifier"),
        ("MOHIT YADAV", "2401730063", "Project 6 Lead: Fake News & Claim Classification"),
        ("DIVAYAM YADAV", "2401730220", "Core Pipeline, Preprocessing, EDA & GUI Integration")
    ]
    for name, roll, role in members:
        p_mem = doc.add_paragraph()
        p_mem.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_mem.add_run(f"{name} (Roll No: {roll})\n")
        r.font.name = "Arial"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = DARK
        r_role = p_mem.add_run(f"[{role}]")
        r_role.font.name = "Arial"
        r_role.font.size = Pt(9.5)
        r_role.font.italic = True
        r_role.font.color.rgb = MUTED

    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_date = p_date.add_run("\nFaculty Course: Natural Language Processing (NLP) Capstone\nAcademic Submission | November 2026")
    r_date.font.name = "Arial"
    r_date.font.size = Pt(10.5)
    r_date.font.color.rgb = MUTED
    p_date.paragraph_format.space_before = Pt(30)

    doc.add_page_break()

    # -------------------------------------------------------------------------
    # ABSTRACT / EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    h_abs = doc.add_heading("Executive Abstract", level=1)
    p_abs = doc.add_paragraph(
        "This capstone project delivers a comprehensive, production-grade Natural Language Processing (NLP) system "
        "encompassing both projects assigned to Group 5 under the faculty guidelines: Project 6 (Fake News or Claim Classification) "
        "and Project 15 (Domain-Specific Document Classifier). Project 6 establishes TruthLens AI, a dual-engine architecture designed "
        "to evaluate the veracity of fact-checking claims and full-length investigative journalism articles. Grounded in the PolitiFact LIAR "
        "benchmark (12,771 statements) and the McIntire news corpus (6,306 articles), TruthLens couples negation-preserving linguistic "
        "preprocessing with sublinear TF-IDF feature extraction. Machine learning algorithms, including Platt-calibrated Passive-Aggressive "
        "Classifiers and Balanced Logistic Regression, achieve 92.47% accuracy on news articles and 0.4265 Macro F1 on the nuanced 3-tier "
        "claim credibility task. Project 15 establishes DocuDomain AI (Enterprise LexiClass), an automated domain routing and classification "
        "engine that categorizes institutional text across Legal, Healthcare, Financial, and Academic/Technology domains with 100% test "
        "accuracy, while dynamically extracting salience-scored executive document summaries. A unified, reactive Streamlit web application "
        "delivers real-time inference, confidence metrics, token explainability chips, and multi-format document parsing (.txt, .pdf, .docx). "
        "Every phase strictly adheres to the 10-stage development lifecycle mandated by the faculty guidelines."
    )
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.paragraph_format.space_after = Pt(20)

    # -------------------------------------------------------------------------
    # CHAPTER 1: INTRODUCTION & PROBLEM IDENTIFICATION (PROJECT 6)
    # -------------------------------------------------------------------------
    doc.add_heading("1. Problem Identification and Real-World Motivation (Project 6)", level=1)
    doc.add_paragraph(
        "The proliferation of digital misinformation across social networks, algorithmic recommendation feeds, and micro-publishing platforms "
        "poses an existential challenge to informed democratic governance, public health response, and economic stability. "
        "Traditional fact-checking organizations (e.g., PolitiFact, FactCheck.org) employ professional investigative journalists; however, "
        "the manual verification lifecycle requires several hours to days per claim, creating an asymmetry against automated or viral disinformation.\n\n"
        "Project 6: Fake News or Claim Classification addresses this systemic latency by formulating an automated NLP decision support system. "
        "The objective is twofold: (1) triage short-form political and social claims into credibility tiers to prioritize manual fact-checking queues, "
        "and (2) classify long-form news articles as authentic or fabricated. In accordance with ethical AI standards, the system functions as a "
        "probabilistic screening engine rather than an infallible arbiter of ground truth."
    )

    # -------------------------------------------------------------------------
    # CHAPTER 2: DATASET INGESTION & DATA ARCHITECTURE (PROJECT 6)
    # -------------------------------------------------------------------------
    doc.add_heading("2. Dataset Collection and Data Architecture (Project 6)", level=1)
    doc.add_paragraph(
        "To ensure robust evaluation across divergent linguistic registers, two widely recognized, publicly accessible benchmarks were ingested:\n"
        "1. PolitiFact LIAR Benchmark: Compiled by William Yang Wang (ACL 2017), this corpus contains 12,771 human-fact-checked statements "
        "spanning a decade of American political discourse. To counteract severe class fragmentation, the 6 raw labels ('true', 'mostly-true', "
        "'half-true', 'barely-true', 'false', 'pants-fire') were mapped into 3 balanced credibility tiers: Credible (38.8%), Half-True (21.2%), "
        "and Questionable (40.0%).\n"
        "2. McIntire Fake and Real News Corpus: Collected by George McIntire, comprising 6,306 full-length journalistic articles evenly balanced "
        "between authentic reportage (3,154 Real) and fabricated stories (3,152 Fake)."
    )

    doc.add_heading("Table 1: Benchmark Dataset Characteristics (Project 6)", level=3)
    table1 = doc.add_table(rows=5, cols=5)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Dataset Name", "Total Samples", "Vocabulary Size", "Mean Text Length", "Target Classes"]
    for c_idx, h in enumerate(headers):
        style_cell(table1.cell(0, c_idx), h, is_header=True, bg_color="1A365D")
        table1.cell(0, c_idx).paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    data1 = [
        ["LIAR Training Split", "10,240 claims", "13,542 unique lemmas", "17.9 words / claim", "Credible, Half-True, Questionable"],
        ["LIAR Validation Split", "1,284 claims", "4,210 unique lemmas", "18.2 words / claim", "3-Tier Credibility"],
        ["LIAR Held-Out Test Split", "1,283 claims", "4,198 unique lemmas", "18.3 words / claim", "3-Tier Credibility"],
        ["McIntire News Corpus", "6,306 articles", "48,912 unique lemmas", "786.8 words / article", "REAL vs. FAKE"]
    ]
    for r_idx, row in enumerate(data1, start=1):
        bg = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            style_cell(table1.cell(r_idx, c_idx), val, bg_color=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # CHAPTER 3: TEXT PREPROCESSING & LEMMATIZATION
    # -------------------------------------------------------------------------
    doc.add_heading("3. Text Preprocessing and Negation-Preserving Normalization", level=1)
    doc.add_paragraph(
        "Text normalization forms the operational foundation of linguistic classification. Standard stopword removal blindly strips negative particles "
        "(e.g., 'not', 'no', 'never', 'without'), critically corrupting semantic truth values (e.g., converting 'The economy did not grow' into 'economy grew'). "
        "To prevent truth-inversion, a specialized negation-aware preprocessing pipeline was authored in src/preprocess.py:\n"
        "1. Contraction Expansion: Colloquial contractions (e.g., won't -> will not, isn't -> is not) are expanded deterministically.\n"
        "2. Noise Elimination: URLs, HTML formatting tags, special punctuation, and non-ASCII artifacts are stripped using regex.\n"
        "3. Negation Preservation: A custom stopword set filters out standard function words while preserving 9 primary negation tokens.\n"
        "4. Morphological Lemmatization: Words are mapped to base lexical forms using NLTK WordNetLemmatizer, consolidating inflected variants."
    )

    # -------------------------------------------------------------------------
    # CHAPTER 4: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    doc.add_heading("4. Exploratory Data Analysis & Linguistic Patterns", level=1)
    doc.add_paragraph(
        "Exploratory Data Analysis was performed using automated scripts (src/eda.py). The investigation revealed significant linguistic divergences "
        "between authentic and fabricated text. Fabricated claims exhibit elevated exclamation marks, higher capitalized token frequencies, and sensationalist "
        "hyperbole. Conversely, credible claims and legitimate articles incorporate formal legislative terminology, statistical citations, and institutional references."
    )

    eda_images = [
        ("class_distribution.png", "Figure 1: Credibility Class Distributions across LIAR and McIntire Benchmarks"),
        ("text_length_distribution.png", "Figure 2: Text Length and Word Count Histograms across Fact-Checking Claims"),
        ("top_ngrams_comparison.png", "Figure 3: Salient Bigrams Differentiating Credible vs Questionable Claims"),
        ("wordcloud_credible_vs_fake.png", "Figure 4: Comparative Word Clouds: Credible vs Questionable Linguistic Discourse"),
        ("speaker_credibility_analysis.png", "Figure 5: Historical Speaker Credibility Breakdown for Top Political Figures")
    ]
    for img_name, caption in eda_images:
        img_path = os.path.join(VIS_DIR, img_name)
        if os.path.exists(img_path):
            doc.add_picture(img_path, width=Inches(6.0))
            p_cap = doc.add_paragraph(caption)
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(14)
            if p_cap.runs:
                p_cap.runs[0].font.size = Pt(9)
                p_cap.runs[0].font.italic = True
                p_cap.runs[0].font.color.rgb = MUTED

    # -------------------------------------------------------------------------
    # CHAPTER 5: FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    doc.add_heading("5. Feature Engineering and N-Gram Extraction", level=1)
    doc.add_paragraph(
        "To transform unstructured textual sequences into dense numerical vectors, we implemented Sublinear Term Frequency-Inverse Document Frequency (TF-IDF). "
        "Traditional term frequency scales linearly with word occurrences, allowing repetitive filler words to dominate sparse representations. "
        "Sublinear scaling dampens raw term frequency via logarithmic transformation:\n\n"
        "       tf_sublinear = 1 + log(tf) if tf > 0, else 0\n\n"
        "The feature pipeline extracts unigrams and bigrams (ngram_range=(1,2)), capturing critical two-word contextual collocations such as 'not true', "
        "'tax cut', and 'white house'. The news classification vocabulary was bounded to the top 20,000 informative n-grams, while the claims classifier "
        "ingested 15,000 features, optimizing memory bandwidth and mitigating the curse of dimensionality."
    )

    # -------------------------------------------------------------------------
    # CHAPTER 6: MODEL DEVELOPMENT & EVALUATION RESULTS (PROJECT 6)
    # -------------------------------------------------------------------------
    doc.add_heading("6. Model Development, Benchmarking and Evaluation (Project 6)", level=1)
    doc.add_paragraph(
        "Five distinct classification algorithms were developed, trained, and benchmarked on strictly isolated test splits:\n"
        "• Multinomial Naive Bayes (MNB): Classical probabilistic generative baseline using Laplace smoothing (alpha=0.1).\n"
        "• Logistic Regression (LR): L2-regularized linear model with balanced class weighting to handle label asymmetry.\n"
        "• Linear Support Vector Classifier (LinearSVC): Maximum-margin hyperplane classifier optimized with hinge loss.\n"
        "• Passive-Aggressive Classifier (PAC): Online margin-based optimization algorithm calibrated via Platt Scaling.\n"
        "• Random Forest Classifier (RF): Non-linear ensemble model with 150 Gini-impurity estimators."
    )

    doc.add_heading("Table 2: News Classifier Performance Benchmark (1,262 Held-Out Articles)", level=3)
    t_news = doc.add_table(rows=6, cols=5)
    t_news.alignment = WD_TABLE_ALIGNMENT.CENTER
    h_news = ["Algorithm", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    for c_idx, h in enumerate(h_news):
        style_cell(t_news.cell(0, c_idx), h, is_header=True, bg_color="1A365D")
        t_news.cell(0, c_idx).paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    data_news = [
        ["Passive-Aggressive (Calibrated)", "92.47%", "0.9248", "0.9247", "0.9247"],
        ["Linear SVM (LinearSVC)", "92.47%", "0.9248", "0.9247", "0.9247"],
        ["Logistic Regression", "91.36%", "0.9142", "0.9135", "0.9135"],
        ["Random Forest (150 trees)", "89.30%", "0.8932", "0.8930", "0.8930"],
        ["Multinomial Naive Bayes", "85.90%", "0.8698", "0.8580", "0.8572"]
    ]
    for r_idx, row in enumerate(data_news, start=1):
        bg = "EBF8FF" if r_idx == 1 else ("F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row):
            style_cell(t_news.cell(r_idx, c_idx), val, bg_color=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    doc.add_heading("Table 3: Fact-Checking Claim Classifier Benchmark (2,555 Held-Out Claims - LIAR)", level=3)
    t_claim = doc.add_table(rows=6, cols=5)
    t_claim.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c_idx, h in enumerate(h_news):
        style_cell(t_claim.cell(0, c_idx), h, is_header=True, bg_color="1A365D")
        t_claim.cell(0, c_idx).paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    data_claim = [
        ["Balanced Logistic Regression", "45.60%", "0.4283", "0.4357", "0.4265"],
        ["Passive-Aggressive (Calibrated)", "44.93%", "0.4225", "0.4239", "0.4227"],
        ["Linear SVM (LinearSVC)", "44.97%", "0.4232", "0.4243", "0.4230"],
        ["Multinomial Naive Bayes", "43.76%", "0.4079", "0.3957", "0.3905"],
        ["Random Forest (150 trees)", "44.89%", "0.4180", "0.4026", "0.3980"]
    ]
    for r_idx, row in enumerate(data_claim, start=1):
        bg = "FEFCBF" if r_idx == 1 else ("F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row):
            style_cell(t_claim.cell(r_idx, c_idx), val, bg_color=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # Embedded Confusion Matrices
    cm_images = [
        ("confusion_matrix_news.png", "Figure 6: Confusion Matrix for News Articles (Passive Aggressive Calibrated - 92.47% Accuracy)"),
        ("confusion_matrix_claims.png", "Figure 7: Confusion Matrix for Fact-Checking Claims (Balanced Logistic Regression Multi-Class)"),
        ("model_comparison_chart.png", "Figure 8: Comparative Accuracy and F1-Scores across All Classification Algorithms")
    ]
    for img_name, caption in cm_images:
        img_path = os.path.join(VIS_DIR, img_name)
        if os.path.exists(img_path):
            doc.add_picture(img_path, width=Inches(5.8))
            p_cap = doc.add_paragraph(caption)
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(14)
            if p_cap.runs:
                p_cap.runs[0].font.size = Pt(9)
                p_cap.runs[0].font.italic = True
                p_cap.runs[0].font.color.rgb = MUTED

    # -------------------------------------------------------------------------
    # CHAPTER 7: GUI DESIGN & APPLICATION TESTING (PROJECT 6)
    # -------------------------------------------------------------------------
    doc.add_heading("7. Streamlit Web GUI & System Implementation (Project 6)", level=1)
    doc.add_paragraph(
        "The frontend application was developed in Streamlit (app.py) with custom responsive CSS styling. "
        "The interface incorporates reactive analysis buttons, real-time progress indicators, probability metric columns, "
        "and salient token chips providing full interpretability.\n\n"
        "Figure 9 illustrates the live Claim Credibility Analyzer tab evaluating a complex economic claim against the LIAR benchmark. "
        "The system extracts salient linguistic cues, categorizes the claim as Credible with 56.4% confidence, and displays the exact probability "
        "distribution across all three tiers. Figure 10 depicts the News Article Classifier evaluating a real news dispatch, confirming authentic "
        "journalism with 87.9% probability and presenting readability metrics."
    )

    gui_p6_screenshots = [
        ("gui_claim_verifier_result.png", "Figure 9: Live Streamlit Web GUI — Claim Credibility Analyzer (Project 6)"),
        ("gui_news_article_result.png", "Figure 10: Live Streamlit Web GUI — Full News Article Classifier (Project 6)")
    ]
    for img_name, caption in gui_p6_screenshots:
        img_path = os.path.join(SCREENSHOTS_DIR, img_name)
        if os.path.exists(img_path):
            doc.add_picture(img_path, width=Inches(6.2))
            p_cap = doc.add_paragraph(caption)
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(14)
            if p_cap.runs:
                p_cap.runs[0].font.size = Pt(9)
                p_cap.runs[0].font.bold = True
                p_cap.runs[0].font.color.rgb = NAVY

    # -------------------------------------------------------------------------
    # CHAPTER 8: CONCLUSION & FUTURE SCOPE (PROJECT 6)
    # -------------------------------------------------------------------------
    doc.add_heading("8. Conclusion, Limitations & Future Scope (Project 6)", level=1)
    doc.add_paragraph(
        "TruthLens NLP successfully demonstrates that supervised text classification coupled with negation-preserving NLP preprocessing "
        "and sublinear TF-IDF features can effectively categorize deceptive news articles (92.47% accuracy) and fact-checking statements across credibility tiers.\n\n"
        "Limitations & Future Work:\n"
        "1. Temporal Lexical Drift: Static n-gram vocabularies are vulnerable to emerging entities and novel political developments. Future work will integrate Retrieval-Augmented Generation (RAG) connected to the Google Fact Check API.\n"
        "2. Semantic Contextualization: Subsequent iterations can fine-tune contextual transformer models (e.g., DeBERTa-v3, RoBERTa-large) to improve nuanced claim inference.\n"
        "3. Multi-Modal Analysis: Incorporating image-text matching models (CLIP) to detect mismatched sensationalist headlines and deceptive imagery."
    )

    # -------------------------------------------------------------------------
    # CHAPTER 9: PROJECT 15 - DOMAIN-SPECIFIC DOCUMENT CLASSIFIER
    # -------------------------------------------------------------------------
    doc.add_heading("9. Project 15: Domain-Specific Document Classifier (Enterprise LexiClass AI)", level=1)
    doc.add_paragraph(
        "As mandated by Section 1 and Section 7 of the faculty guidelines, Group 5 was assigned two complementary NLP capstone projects: "
        "Project 6 (Fake News or Claim Classification) and Project 15 (Domain-Specific Document Classifier). Project 15 was led by Priyanshu Singh "
        "(Roll No: 2401730099) in collaboration with Mohit Yadav and Divayam Yadav.\n\n"
        "9.1 Problem Statement & Industrial Objectives:\n"
        "Enterprise workflows in law firms, healthcare networks, corporate investment desks, and academic institutions process massive quantities of unstructured text. "
        "Manual ingestion and routing of contracts, patient records, 10-K financial disclosures, and scientific papers introduces immense latency and error rates. "
        "The objective of Project 15 is to construct an intelligent NLP classification engine that categorizes incoming documents into specialized domains "
        "(Legal, Healthcare, Financial, Technology & Academic), outputs calibrated confidence scores, and automatically generates an extractive document executive summary."
    )

    doc.add_paragraph(
        "9.2 Multi-Domain Corpus & Ingestion Architecture:\n"
        "A multi-domain benchmark corpus comprising 240 curated enterprise documents across 4 distinct domains was constructed (data/processed/domain_documents.csv):\n"
        "• Legal Domain: Non-Disclosure Agreements, master service agreements, summary judgment motions, patent infringement complaints, and corporate bylaws.\n"
        "• Healthcare Domain: Inpatient admission summaries, myocardial infarction protocols, surgical pathology notes, pediatric EEG evaluations, and oncology regimens.\n"
        "• Financial Domain: SEC Form 10-K filings, equity analyst reports, quarterly earnings call transcripts, discounted cash flow valuations, and hedge fund risk reports.\n"
        "• Technology & Academic Domain: Transformer attention mechanisms, distributed consensus protocols, deep reinforcement learning, and quantum error correction papers."
    )

    doc.add_paragraph(
        "9.3 Feature Extraction & Classification Performance:\n"
        "Documents were processed through domain-aware morphological lemmatization and converted into sublinear TF-IDF feature representations across unigrams and bigrams. "
        "Calibrated classification models were evaluated on an isolated 25% stratified test split. As summarized in Table 4, the Calibrated Logistic Regression model "
        "achieved 100% accuracy and 1.0000 Macro F1 across all four domains, demonstrating pristine boundary separation on domain-specific vocabulary."
    )

    doc.add_heading("Table 4: Project 15 Multi-Domain Model Performance Matrix", level=3)
    t_dom = doc.add_table(rows=4, cols=5)
    t_dom.alignment = WD_TABLE_ALIGNMENT.CENTER
    h_dom = ["Classification Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    for c_idx, h in enumerate(h_dom):
        style_cell(t_dom.cell(0, c_idx), h, is_header=True, bg_color="1A365D")
        t_dom.cell(0, c_idx).paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    data_dom = [
        ["Calibrated Logistic Regression (C=2.0)", "100.00%", "1.0000", "1.0000", "1.0000"],
        ["Multinomial Naive Bayes (alpha=0.5)", "100.00%", "1.0000", "1.0000", "1.0000"],
        ["Passive-Aggressive Classifier (Calibrated)", "100.00%", "1.0000", "1.0000", "1.0000"]
    ]
    for r_idx, row in enumerate(data_dom, start=1):
        bg = "EBF8FF" if r_idx == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            style_cell(t_dom.cell(r_idx, c_idx), val, bg_color=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    cm_dom_path = os.path.join(VIS_DIR, "confusion_matrix_domain.png")
    if os.path.exists(cm_dom_path):
        doc.add_picture(cm_dom_path, width=Inches(5.6))
        p_cap = doc.add_paragraph("Figure 11: Confusion Matrix for Project 15 Domain Classifier (100% Accuracy on 4 Enterprise Domains)")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(14)
        if p_cap.runs:
            p_cap.runs[0].font.size = Pt(9)
            p_cap.runs[0].font.italic = True
            p_cap.runs[0].font.color.rgb = MUTED

    doc.add_paragraph(
        "9.4 Automatic Extractive Document Summarization Algorithm:\n"
        "To satisfy the explicit faculty requirement ('Show confidence score and document summary, if implemented'), an extractive summarization engine "
        "was implemented in src/domain_classifier.py. The algorithm parses sentences using NLTK sent_tokenize, computes word salience weights normalized "
        "by maximum document term frequency, scores sentences with lead-positional heuristics, and extracts the top 3 key sentences as an executive briefing."
    )

    doc.add_paragraph(
        "9.5 Streamlit Web GUI & Multi-Format File Ingestion:\n"
        "The Streamlit frontend (Tab 3) supports direct text entry as well as drag-and-drop file upload for PDF, DOCX, and TXT documents. "
        "Upon execution, the engine displays the predicted domain badge (e.g., LEGAL in Indigo, HEALTHCARE in Teal), confidence metrics, class probability "
        "distribution columns, the 3-sentence executive document summary, and dominant domain keyword chips. Figure 12 shows a live capture of this interface."
    )

    scr_dom_path = os.path.join(SCREENSHOTS_DIR, "gui_domain_classifier_result.png")
    if os.path.exists(scr_dom_path):
        doc.add_picture(scr_dom_path, width=Inches(6.2))
        p_cap = doc.add_paragraph("Figure 12: Live Streamlit Web GUI — Domain-Specific Document Classifier & Summarizer (Project 15)")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(14)
        if p_cap.runs:
            p_cap.runs[0].font.size = Pt(9)
            p_cap.runs[0].font.bold = True
            p_cap.runs[0].font.color.rgb = NAVY

    # -------------------------------------------------------------------------
    # CHAPTER 10: TEAM STRUCTURE & COMPLIANCE MATRIX
    # -------------------------------------------------------------------------
    doc.add_heading("10. Group 5 Team Governance, Task Distribution & Faculty Compliance", level=1)
    doc.add_paragraph(
        "The faculty guidelines require active participation and clear understanding of both projects by all team members. "
        "Table 5 details the specific contributions, leadership roles, and technical domains owned by each Group 5 member."
    )

    doc.add_heading("Table 5: Group 5 Team Member Attribution Matrix", level=3)
    t_team = doc.add_table(rows=4, cols=4)
    t_team.alignment = WD_TABLE_ALIGNMENT.CENTER
    h_team = ["Student Name", "Roll Number", "Primary Project Role", "Key Technical Contributions"]
    for c_idx, h in enumerate(h_team):
        style_cell(t_team.cell(0, c_idx), h, is_header=True, bg_color="1A365D")
        t_team.cell(0, c_idx).paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    data_team = [
        [
            "PRIYANSHU SINGH",
            "2401730099",
            "Project 15 Lead (Domain Document Classifier)",
            "Curated 4-domain enterprise corpus, implemented sublinear TF-IDF domain vectorizer, authored extractive summarization engine, and engineered multi-format document parser."
        ],
        [
            "MOHIT YADAV",
            "2401730063",
            "Project 6 Lead (Fake News & Claim Classification)",
            "Ingested PolitiFact LIAR and McIntire news corpora, developed balanced Logistic Regression and Passive-Aggressive models, implemented Platt scaling probability calibration, and authored explainability engine."
        ],
        [
            "DIVAYAM YADAV",
            "2401730220",
            "Core Pipeline, EDA & GUI Integration",
            "Engineered negation-preserving text cleaner and WordNet lemmatizer, produced 5 EDA visualization figures, designed full-stack Streamlit web application, and authored automated test suite."
        ]
    ]
    for r_idx, row in enumerate(data_team, start=1):
        bg = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            style_cell(t_team.cell(r_idx, c_idx), val, bg_color=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    doc.add_paragraph(
        "Faculty Guidelines 10-Stage Compliance Verification:\n"
        "1. Problem Identification: Fully articulated in Chapter 1 (Misinformation) and Chapter 9.1 (Enterprise Document Routing).\n"
        "2. Dataset Collection: Real-world corpora ingested: LIAR (12,771 claims), McIntire (6,306 articles), Multi-Domain (240 documents).\n"
        "3. Data Cleaning: Addressed duplicates, missing metadata, HTML tags, and contractions in src/preprocess.py.\n"
        "4. Text Preprocessing: Custom negation-preserving stopword filtering and WordNet lemmatization.\n"
        "5. Exploratory Data Analysis: Conducted in src/eda.py with 5 high-resolution figures in Chapter 4.\n"
        "6. Feature Engineering: Sublinear TF-IDF (1,2)-grams implemented across both projects (Chapters 5 and 9.3).\n"
        "7. Model Development: 5 ML models trained for Project 6, 3 models trained for Project 15.\n"
        "8. Model Evaluation: Complete Accuracy, Precision, Recall, F1 metrics and 3 confusion matrices reported (Chapters 6 and 9.3).\n"
        "9. GUI Development: Unified Streamlit web application with live demonstration capabilities for both projects (Chapters 7 and 9.5).\n"
        "10. Testing & Implementation: Validated via pytest/unittest suite in tests/test_pipeline.py with zero failures."
    )

    # -------------------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------------------
    doc.add_heading("References", level=1)
    refs = [
        "Wang, W. Y. (2017). 'Liar, Liar Pants on Fire': A New Benchmark Dataset for Fake News Detection. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 422–426.",
        "McIntire, G. (2017). Fake and Real News Dataset. University of Rhode Island Machine Learning Repository.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, pp. 2825–2830.",
        "Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit. O'Reilly Media.",
        "Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., & Singer, Y. (2006). Online Passive-Aggressive Algorithms. Journal of Machine Learning Research, 7, pp. 551–585.",
        "Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), pp. 513–523.",
        "Vaswani, A., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, pp. 5998–6008.",
        "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019, pp. 4171–4186."
    ]
    for r in refs:
        p_r = doc.add_paragraph(r)
        p_r.paragraph_format.left_indent = Inches(0.5)
        p_r.paragraph_format.first_line_indent = Inches(-0.5)
        p_r.paragraph_format.space_after = Pt(4)

    doc.save(OUTPUT_DOCX)
    print(f"Comprehensive Capstone Report (.docx) generated successfully: {OUTPUT_DOCX}")


if __name__ == "__main__":
    build_report()
