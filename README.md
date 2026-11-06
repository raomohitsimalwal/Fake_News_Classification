# 🛡️ TruthLens NLP: Fake News & Claim Credibility Classification System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![NLP Benchmark](https://img.shields.io/badge/Benchmark-PolitiFact%20LIAR-brightgreen.svg)](https://github.com/thiagorainmaker77/liar_dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-ready Natural Language Processing (NLP) Capstone application developed to classify short fact-checking claims and full-length news articles into empirical credibility tiers. Built entirely from scratch using authentic public benchmark datasets, rigorous feature engineering, multi-algorithm machine learning benchmarks, and an explainable Streamlit interactive web interface.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement & Objectives](#-problem-statement--objectives)
3. [Dataset Source & Description](#-dataset-source--description)
4. [Technologies & Libraries Used](#-technologies--libraries-used)
5. [NLP Preprocessing Techniques](#-nlp-preprocessing-techniques)
6. [Feature Extraction Methods](#-feature-extraction-methods)
7. [Machine Learning Algorithms](#-machine-learning-algorithms)
8. [Actual Model Evaluation Results](#-actual-model-evaluation-results)
9. [Project Directory Structure](#-project-directory-structure)
10. [Installation & Setup Instructions](#-installation--setup-instructions)
11. [Execution Instructions](#-execution-instructions)
12. [Streamlit GUI Features & Walkthrough](#-streamlit-gui-features--walkthrough)
13. [Limitations & Future Scope](#-limitations--future-scope)
14. [Academic & Ethical Transparency Disclaimer](#-academic--ethical-transparency-disclaimer)

---

## 📖 Project Overview

The internet and decentralized social platforms have dramatically lowered the barrier for publishing and disseminating unverified information. While computational text classification cannot replace human investigative journalism, NLP models can rapidly screen digital text, categorize claims into statistical credibility categories, and flag deceptive rhetorical markers.

**TruthLens NLP** is a dual-engine credibility classification system:
1. **Fact-Checking Claim Analyzer:** Evaluates short public statements across three credibility categories (`Credible`, `Half-True`, `Questionable`) based on the gold-standard **PolitiFact LIAR Benchmark** (12,771 claims).
2. **Full News Article Verifier:** Detects structural deception in long-form journalistic writing, classifying articles as `REAL` or `FAKE` with **92.47% accuracy**.

---

## 🎯 Problem Statement & Objectives

### Problem Statement
Online discourse suffers from pervasive misinformation, including fabrications, out-of-context distortions, and selective half-truths. Detecting deception manually across millions of daily posts is intractable. This project addresses the challenge of building an automated, interpretable, and reproducible text classification pipeline that predicts the credibility of statements based on historical, human-annotated fact-checking archives.

### Project Objectives
- Ingest and clean authentic public NLP datasets without using synthetic or fabricated records.
- Implement specialized NLP preprocessing including negation-aware stopword filtering and WordNet lemmatization.
- Conduct exploratory data analysis (EDA) to expose lexical discriminators, n-gram patterns, and speaker tendencies.
- Extract high-dimensional n-gram representations using Sublinear TF-IDF scaling.
- Train and compare 5 distinct classification algorithms using stratified 80/20 train-test splits.
- Evaluate models using Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score, and Confusion Matrices.
- Build a modern, responsive, and explainable Streamlit web GUI.
- Clearly present predictions as statistical inferences rather than infallible claims of absolute ground truth.

---

## 📊 Dataset Source & Description

The system uses two authentic, peer-reviewed public datasets:

### 1. PolitiFact LIAR Benchmark (William Yang Wang, ACL 2017)
- **Source:** PolitiFact fact-checking portal ([Dataset Paper](https://aclanthology.org/P17-2067.pdf)).
- **Total Records:** 12,771 fact-checked statements.
- **Features:** Statement text, Speaker, Job title, State info, Party affiliation, Historical truth counts, Context/venue.
- **Original 6 Labels:** `true` (2,050), `mostly-true` (2,453), `half-true` (2,624), `barely-true` (2,103), `false` (2,494), `pants-fire` (1,047).
- **Consolidated Credibility Tiers:**
  - **Credible (4,503 claims):** Comprising `true` and `mostly-true` statements.
  - **Half-True (2,624 claims):** Statements that are partially accurate but lack critical context or nuance.
  - **Questionable (5,644 claims):** Comprising `barely-true`, `false`, and `pants-fire` claims.

### 2. Fake or Real News Benchmark (George McIntire / Lutz Hamel)
- **Source:** University of Rhode Island / McIntire Fake News Corpus.
- **Total Records:** 6,306 complete news articles.
- **Class Balance:** Perfectly balanced (3,154 REAL, 3,152 FAKE).
- **Features:** Article title, full article text, veracity label.

---

## 🛠️ Technologies & Libraries Used

| Category | Tool / Library | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11+ | Core programming runtime |
| **NLP & Preprocessing** | NLTK | 3.10+ | Tokenization, lemmatization, stopword corpora |
| **Machine Learning** | Scikit-Learn | 1.8+ | Classifiers, TF-IDF vectorization, calibration, metrics |
| **Data Manipulation** | Pandas & NumPy | 2.1+ / 1.26+ | Matrix operations, tabular dataframe processing |
| **Data Visualization** | Matplotlib & Seaborn | 3.10+ / 0.13+ | Publication-grade confusion matrices and EDA charts |
| **Word Cloud** | WordCloud | 1.9+ | Lexical visualization of prominent veracity tokens |
| **Model Persistence** | Joblib | 1.5+ | Serializing and loading trained models and vectorizers |
| **Web GUI** | Streamlit | 1.40+ | Modern reactive frontend and interactive dashboard |

---

## 🔬 NLP Preprocessing Techniques

Text preprocessing follows a modular, five-stage pipeline implemented in [`src/preprocess.py`](src/preprocess.py):

1. **Noise Removal:**
   - Stripping HTTP/HTTPS URLs and domain hyperlinks using regular expressions.
   - Eliminating HTML tags (`<.*?>`) and residual markup entities.
   - Removing special symbols and non-alphabetical punctuation.
2. **Contraction Expansion:**
   - Expanding colloquial contractions (e.g. `can't` -> `cannot`, `won't` -> `will not`, `it's` -> `it is`) to preserve semantic completeness.
3. **Case Normalization:**
   - Converting all input text to lowercase.
4. **Negation-Preserving Stopword Filtering:**
   - Standard NLP stopword removal indiscriminately strips negation words (`not`, `no`, `never`, `cannot`, `without`). In fact-checking, negation fundamentally reverses the truth value of a claim!
   - *Example:* "The candidate did **not** accept foreign donations" vs "The candidate did accept..."
   - TruthLens NLP uses a custom negation-aware stopword dictionary that preserves all 9 primary English negation tokens.
5. **WordNet Lemmatization:**
   - Reducing inflected word forms to their dictionary root lemmas (`reporters` -> `reporter`, `denies` -> `deny`, `increased` -> `increase`) using `nltk.stem.WordNetLemmatizer`.

---

## 📐 Feature Extraction Methods

Features are extracted using high-dimensional **Term Frequency - Inverse Document Frequency (TF-IDF)** vectorization:

- **N-gram Range `(1, 2)`:** Extracts both individual words (unigrams) and adjacent word pairs (bigrams), allowing the model to capture compound phrases such as `health care`, `tax cut`, `white house`, and `social security`.
- **Sublinear Term Frequency Scaling:** Applies logarithmic scaling `tf = 1 + log(tf)` to prevent frequent repetitive words from disproportionately skewing document vectors.
- **Vocabulary Sizing:**
  - News Vectorizer: `max_features = 12,000`, `min_df = 3`.
  - Claim Vectorizer: `max_features = 10,000`, `min_df = 2`.
- **Unicode Normalization:** Strips accidental diacritics and accents.

---

## 🤖 Machine Learning Algorithms

Five diverse classification algorithms were trained and comparatively benchmarked:

1. **Multinomial Naive Bayes (`MultinomialNB`):** Generative probabilistic classifier modeling token presence under class-conditional independence.
2. **Logistic Regression (`LogisticRegression`):** Linear model optimizing cross-entropy loss with L2 regularization and balanced class weights, providing calibrated probability log-odds.
3. **Passive-Aggressive Classifier (`PassiveAggressiveClassifier`):** Margin-based online learning algorithm renowned in fake news literature for updating weights aggressively upon misclassification while remaining passive on correct predictions.
4. **Linear Support Vector Classifier (`LinearSVC` wrapped in `CalibratedClassifierCV`):** Maximum-margin hyper-plane separation with sigmoid probability calibration.
5. **Random Forest Classifier (`RandomForestClassifier`):** Bagging ensemble of 100 decision trees evaluating non-linear feature interactions.

---

## 📈 Actual Model Evaluation Results

All models were evaluated on strictly isolated, held-out test splits (20% of data, zero data leakage).

### 1. News Article Verifier Benchmark (1,262 Unseen Articles)
| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Passive Aggressive (Calibrated)** | **92.47%** | **0.9248** | **0.9247** | **0.9247** | 🏆 **Best Performing** |
| Linear SVM (Calibrated) | 92.23% | 0.9224 | 0.9223 | 0.9223 | Strong Contender |
| Logistic Regression | 91.68% | 0.9170 | 0.9168 | 0.9168 | Highly Interpretable |
| Multinomial Naive Bayes | 89.30% | 0.8956 | 0.8930 | 0.8929 | Fast Baseline |
| Random Forest (100 Trees) | 88.27% | 0.8850 | 0.8827 | 0.8826 | Ensemble Baseline |

### 2. Claim Credibility Benchmark (2,555 Unseen Fact-Checking Claims)
| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **45.60%** | **0.4287** | **0.4267** | **0.4265** | 🏆 **Best Macro F1** |
| Multinomial Naive Bayes | 50.72% | 0.4549 | 0.4248 | 0.3955 | Skewed toward Majority |
| Passive Aggressive | 43.01% | 0.3939 | 0.3935 | 0.3937 | Robust Online |
| Linear SVM (Calibrated) | 49.67% | 0.3275 | 0.4034 | 0.3550 | Margin Based |
| Random Forest | 48.49% | 0.3227 | 0.3845 | 0.3257 | Sparse Underfitting |

> **Academic Insight:** Fact-checking claim classification on the PolitiFact LIAR benchmark is notoriously challenging due to claim brevity (mean ~18 words) and real-world political ambiguity. In published literature (William Yang Wang, ACL 2017), fine-grained multi-class accuracy typically ranges between 27% and 48%. Conversely, full news articles achieve >92% accuracy due to abundant lexical signals.

---

## 📁 Project Directory Structure

```
Fake_News_Classification/
├── app.py                      # Main modern Streamlit Web Application
├── requirements.txt            # Python environment dependencies
├── README.md                   # Comprehensive project documentation
├── presentation_outline.md     # 10-slide Capstone presentation outline
├── run_app.bat                 # 1-click launcher script for Windows
├── data/
│   ├── raw/                    # Raw downloaded benchmark datasets
│   │   ├── liar_train.tsv
│   │   ├── liar_test.tsv
│   │   ├── liar_valid.tsv
│   │   └── fake_or_real_news.csv
│   └── processed/              # Cleaned, preprocessed datasets
│       ├── liar_processed.csv
│       └── news_processed.csv
├── models/
│   ├── news_classifier.joblib  # Serialized best news classifier
│   ├── news_vectorizer.joblib  # Serialized news TF-IDF vectorizer
│   ├── claim_classifier.joblib # Serialized best claim classifier
│   ├── claim_vectorizer.joblib # Serialized claim TF-IDF vectorizer
│   └── evaluation_metrics.json # Full evaluation results & classification reports
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Jupyter Notebook covering EDA
├── src/
│   ├── __init__.py
│   ├── download_data.py        # Automated dataset downloader
│   ├── preprocess.py           # Text cleaning and lemmatization pipeline
│   ├── eda.py                  # Exploratory data analysis & figure generator
│   ├── train_models.py         # Multi-model training and evaluation pipeline
│   └── explainability.py       # Inference and keyword attribution engine
├── tests/
│   └── test_pipeline.py        # Automated unit and integration test suite
└── visualizations/             # High-resolution saved charts & plots
    ├── class_distribution.png
    ├── text_length_distribution.png
    ├── top_ngrams_comparison.png
    ├── wordcloud_credible_vs_fake.png
    ├── speaker_credibility_analysis.png
    ├── confusion_matrix_news.png
    ├── confusion_matrix_claims.png
    ├── model_comparison_chart.png
    └── eda_summary.json
```

---

## 💻 Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, or newer.
- Git (optional, for version control).

### Step 1: Clone or Open the Project
Navigate to the project root directory:
```bash
cd Fake_News_Classification
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Corpora (Automated)
Run the following one-line command to ensure all NLTK tokenizers and lemmatizer dictionaries are cached:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

---

## 🚀 Execution Instructions

### Option A: 1-Click Launch (Windows)
Double-click the `run_app.bat` file in the project folder.

### Option B: Command Line Launch
Run Streamlit directly from your terminal:
```bash
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

### Option C: Retraining Pipeline from Scratch
To re-download datasets, clean text, re-generate EDA figures, and retrain all models:
```bash
# 1. Download raw data
python src/download_data.py

# 2. Preprocess text
python src/preprocess.py

# 3. Generate EDA visualizations
python src/eda.py

# 4. Train, benchmark, and save models
python src/train_models.py

# 5. Run test suite
python tests/test_pipeline.py
```

---

## 🖥️ Streamlit GUI Features & Walkthrough

The web interface is organized into 5 dedicated tabs:

1. **🔍 Tab 1: Fact-Checking Claim Verifier (LIAR)**
   - Text input for public statements with optional Speaker and Subject fields.
   - Preset sample buttons: *Sample Credible Claim*, *Sample Half-True Claim*, *Sample False/Questionable Claim*.
   - **Analyze Claim Credibility** button.
   - Color-coded verdict card (Green for Credible, Yellow for Half-True, Red for Questionable) with confidence score percentage.
   - Interactive probability breakdown bar chart.
   - Influential token chips highlighting keywords that pushed the model toward specific credibility categories.
   - Stylometric metrics: word count, presence of ALL-CAPS, exclamation marks, and numeric specificity.

2. **📰 Tab 2: Full News Article Classifier**
   - Headline and article body text areas.
   - Preset buttons for real journalistic reports (Reuters) and deceptive viral clickbait.
   - **Verify News Article** button.
   - Output card indicating `REAL` or `FAKE` with confidence score.
   - Reading time estimation and lexical diversity (type-token ratio) indicators.

3. **📊 Tab 3: Model Benchmark & Metrics**
   - Tabular comparison of all 5 algorithms on held-out test data.
   - High-resolution confusion matrices for both claim and news article classifiers.
   - Comparative multi-metric bar chart.

4. **📈 Tab 4: Dataset & EDA Explorer**
   - High-level metric counters (total claims, total news articles, mean lengths).
   - Embedded high-res visual figures: Class balance, text length boxplots, top bigram comparisons, word clouds, and speaker credibility distributions.

5. **ℹ️ Tab 5: Methodology & Architecture**
   - Comprehensive walk-through of the 10-stage Capstone development lifecycle.
   - Technical justification for why Passive-Aggressive and Logistic Regression algorithms were selected.
   - Explicit discussion of real-world limitations and ethical guidelines.

---

## ⚠️ Limitations & Future Scope

### Current Limitations
- **Corpus-Bound Vocabulary:** The model is bounded by its training vocabulary. Rapidly emerging political figures or novel scientific terminology may suffer from out-of-vocabulary (OOV) sparsity.
- **Absence of Real-Time Fact Retrieval:** The model evaluates stylistic, rhetorical, and lexical patterns; it does not query live knowledge graphs or search engines to verify whether an event occurred yesterday.
- **Rhetorical Sarcasm & Irony:** Satirical articles from outlets like *The Onion* may mimic authentic journalistic syntax while presenting absurd premises.

### Future Enhancements
1. **Retrieval-Augmented Generation (RAG):** Integrate the classification engine with the Google Fact Check Tools API and Wikipedia search to retrieve corroborating source documents in real time.
2. **Transformer Embeddings:** Fine-tune pre-trained language models (RoBERTa, DeBERTa-v3) to capture contextual semantic representations beyond n-gram TF-IDF.
3. **Multi-Modal Verification:** Incorporate computer vision models (CLIP) to verify whether thumbnail images match news article text.

---

## ⚖️ Academic & Ethical Transparency Disclaimer

> **IMPORTANT:** This software is an educational machine learning demonstrator designed for an academic NLP Capstone project. It produces predictions by identifying statistical patterns in historical training corpora (PolitiFact LIAR and McIntire benchmarks). **It does not determine, nor does it claim to determine, the absolute truth or falsehood of any real-world statement.** Users must always consult verified, independent journalistic and scientific sources before drawing conclusions about controversial claims.

---
*TruthLens NLP Capstone Project | Academic Session 2026 | MIT License*
