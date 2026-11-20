"""
Project 15: Domain-Specific Document Classifier (Enterprise LexiClass AI).
Part of Group 5 NLP Capstone Project Submission.
Team Members:
  - Priyanshu Singh (Roll No: 2401730099) - Project 15 Lead
  - Mohit Yadav (Roll No: 2401730063) - Project 6 Lead
  - Divayam Yadav (Roll No: 2401730220) - Pipeline & UI Integration

Categorizes enterprise and academic documents into 4 primary domains:
  1. Legal (Contracts, NDAs, Litigation, Compliance, Statutory Filings)
  2. Healthcare (Clinical Notes, Pharmacology, Radiology, Pathology Reports)
  3. Financial (10-K Disclosures, Earnings Reports, Equity Research, Audits)
  4. Technology / Academic (Machine Learning Papers, System Architecture, Software Specifications)

Includes:
  - Text preprocessing and normalization
  - TF-IDF feature extraction (unigrams + bigrams)
  - Calibrated Classifier for multi-class probability estimation
  - Automatic extractive document summarization (TextRank / salience scoring)
"""

import os
import re
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# Use locally available NLTK resources directly without downloading
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set(["the", "and", "is", "in", "to", "of", "a", "with", "for", "as", "on", "that", "this", "by", "an", "be", "or"])

def preprocess_domain_text(text: str) -> str:
    """Clean and normalize domain document text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(cleaned)

def extractive_summarize(text: str, max_sentences: int = 3) -> str:
    """
    Extractive document summarization using sentence-level word salience scoring.
    Fulfills the faculty requirement: 'Show confidence score and document summary, if implemented'.
    """
    if not isinstance(text, str) or len(text.strip()) < 50:
        return text.strip()
    
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) if w.lower() not in stop_words]
    if not words:
        return " ".join(sentences[:max_sentences])
    
    # Word frequency map
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    max_f = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] /= max_f
        
    # Score sentences
    sentence_scores = []
    for idx, s in enumerate(sentences):
        s_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', s)]
        score = sum(freq.get(w, 0) for w in s_words)
        # Position boost for lead sentences
        pos_boost = 1.2 if idx == 0 else (1.1 if idx == 1 else 1.0)
        norm_score = (score / (len(s_words) + 1)) * pos_boost
        sentence_scores.append((norm_score, idx, s))
        
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    top_sentences = sorted(sentence_scores[:max_sentences], key=lambda x: x[1])
    return " ".join(s[2].strip() for s in top_sentences)

def create_domain_corpus():
    """Generate and compile a rich multi-domain corpus for training."""
    csv_path = os.path.join(DATA_DIR, "domain_documents.csv")
    
    # Domain document archetypes with rich vocabulary
    docs = [
        # LEGAL SAMPLES
        ("Legal", "This Non-Disclosure Agreement (the 'Agreement') is entered into by and between the Disclosing Party and Receiving Party. All proprietary technical designs, trade secrets, software algorithms, customer records, and financial balance sheets disclosed herein shall be maintained in strict confidentiality. Neither party shall disclose Confidential Information to any third party without express written consent. In the event of a breach, equitable relief and injunctive remedies shall be available in court of competent jurisdiction under the laws of the State of Delaware."),
        ("Legal", "The Plaintiff hereby files this Motion for Summary Judgment pursuant to Rule 56 of the Federal Rules of Civil Procedure. There is no genuine dispute as to any material fact, and the moving party is entitled to judgment as a matter of law. Defendant failed to perform contractual obligations under Section 4.2 of the Master Services Agreement, directly causing compensatory damages exceeding statutory limits. Affidavits and depositions submitted herewith substantiate the liability of the respondent."),
        ("Legal", "Terms of Service and License Agreement: By downloading, accessing or utilizing this software suite, the Licensee agrees to be bound by the statutory covenants set forth herein. The Licensor grants a non-exclusive, non-transferable, revocable license to execute the object code solely for internal commercial operations. Reverse engineering, decompilation, and sub-licensing are strictly prohibited under applicable copyright jurisprudence."),
        ("Legal", "Articles of Incorporation and Corporate Bylaws: The corporation shall have perpetual existence and is organized under the General Corporation Law. The authorized capital stock consists of common shares with voting privileges and preferred shares with dividend seniority. The Board of Directors shall possess full governance authority to enact resolutions, amend bylaws, and authorize debt financing subject to shareholder quorum requirements."),
        ("Legal", "Employment Arbitration Clause: Any dispute, claim or controversy arising out of or relating to the employment relationship, termination, severance agreements, or alleged violation of statutory employment protections shall be submitted to mandatory binding arbitration administered by the American Arbitration Association. The arbitrator shall have exclusive authority to resolve jurisdictional issues and award statutory damages."),
        ("Legal", "Patent Infringement Complaint: Plaintiff asserts claims against Defendant for willful infringement of United States Patent No. 9,482,105, entitled 'Distributed Data Processing System'. Defendant has manufactured, offered for sale, and imported infringing telecommunication devices embodying each claim limitation without authority or license from the patent holder."),
        ("Legal", "Commercial Lease Agreement: Landlord hereby leases to Tenant the commercial premises for a term of sixty calendar months. Tenant covenants to pay base monthly rent plus proportionate operating expenses, real estate taxes, and building maintenance. Tenant shall maintain comprehensive general liability insurance indemnifying Landlord against casualty losses and premises liability."),
        ("Legal", "Settlement and Release Agreement: In consideration of the mutual covenants and settlement sum paid by Defendant, Plaintiff unconditionally discharges, acquits, and forever releases Defendant and its officers from all past, present, and future claims, demands, and causes of action arising from the subject automobile incident and associated litigation."),
        ("Legal", "Statutory Compliance Notice: Under federal environmental statutes and the Clean Water Act, industrial operators must obtain National Pollutant Discharge Elimination System permits prior to effluent discharge. Failure to maintain toxicological monitoring records constitutes a misdemeanor punishable by civil penalties and mandatory corrective compliance orders."),
        ("Legal", "Intellectual Property Assignment Agreement: The Assignor hereby irrevocably assigns, transfers, and conveys to Assignee all right, title, and interest in and to all inventions, patent applications, trademarks, works of authorship, and copyright registrations developed during the course of consulting services."),

        # HEALTHCARE / MEDICAL SAMPLES
        ("Healthcare", "Patient Admission & Clinical Assessment: A 64-year-old male presents to the Emergency Department with acute substernal chest discomfort radiating to the left arm, diaphoresis, and shortness of breath. Vital signs: blood pressure 168/94 mmHg, heart rate 104 bpm, oxygen saturation 93% on room air. 12-lead electrocardiogram demonstrates 2mm ST-segment elevation in leads V2 through V4 indicative of acute anterior myocardial infarction. Administered sublingual nitroglycerin, 325mg aspirin, and initiated urgent cardiac catheterization protocol."),
        ("Healthcare", "Diagnostic Radiology Report: CT scan of the abdomen and pelvis performed with intravenous contrast. Examination reveals a 4.2 cm hypoattenuating lesion within hepatic segment VI, with peripheral nodular arterial enhancement and gradual centripetal filling on delayed phases, consistent with hepatic hemangioma. The pancreas, spleen, and adrenal glands are unremarkable. No free fluid or retroperitoneal lymphadenopathy identified."),
        ("Healthcare", "Clinical Pharmacology & Treatment Protocol: Patient diagnosed with Type 2 Diabetes Mellitus with elevated Glycated Hemoglobin (HbA1c 8.9%). Commenced oral Metformin hydrochloride 500 mg twice daily with meals to improve insulin sensitivity and reduce hepatic gluconeogenesis. Recommended concurrent lifestyle modification, daily glycemic monitoring, and subcutaneous Semaglutide 0.25 mg weekly if target HbA1c below 7.0% is not attained at 12-week follow-up."),
        ("Healthcare", "Surgical Pathology Consultation: Specimen designated 'Right lower lobe lung wedge resection'. Microscopic examination demonstrates invasive adenocarcinoma with acinar and papillary architectural patterns. Immunohistochemistry reveals strong cytoplasmic positivity for TTF-1 and Napsin-A, negative for p40 and CK5/6. Surgical margins are clear of neoplastic cells. Recommended molecular profiling for EGFR mutations and ALK rearrangements."),
        ("Healthcare", "Pediatric Neurological Assessment: An 8-year-old female evaluated for recurring episodes of unresponsiveness and rhythmic eyelid fluttering lasting 5 to 10 seconds. Electroencephalogram (EEG) demonstrates generalized 3 Hz spike-and-wave discharges activated by hyperventilation, pathognomonic for childhood absence epilepsy. Commenced Ethosuximide therapy with therapeutic drug level monitoring to achieve seizure freedom."),
        ("Healthcare", "Oncology Treatment Summary: Patient undergoing adjuvant chemotherapy for stage IIB invasive ductal carcinoma of the breast following lumpectomy and sentinel lymph node biopsy. Regimen consists of 4 cycles of Doxorubicin and Cyclophosphamide followed by weekly Paclitaxel. Regular complete blood counts monitored for chemotherapy-induced neutropenia and febrile episodes."),
        ("Healthcare", "Infectious Disease Clinical Notes: 42-year-old immunocompromised patient presenting with productive cough, high fever, and bilateral interstitial pulmonary infiltrates. Bronchoalveolar lavage fluid confirms Pneumocystis jirovecii via direct fluorescent antibody staining. Commenced intravenous Trimethoprim-Sulfamethoxazole alongside corticosteroid taper. CD4 count measured at 85 cells/microliter."),
        ("Healthcare", "Orthopedic Operative Note: Preoperative diagnosis: severe bilateral osteoarthritis of the right knee joint. Procedure performed: Right total knee arthroplasty using cemented tricompartmental prosthesis. Femoral and tibial bone cuts executed with precision navigational guides. Adequate patellar tracking and ligamentous balance confirmed prior to closure."),
        ("Healthcare", "Psychiatric Diagnostic Evaluation: Patient meets DSM-5 criteria for Major Depressive Disorder, recurrent, moderate severity, accompanied by generalized anxiety symptoms. Mental status exam shows psychomotor slowing, blunted affect, and sleep architecture disruption with terminal insomnia. Initiated Escitalopram 10mg daily and referred for cognitive behavioral psychotherapy."),
        ("Healthcare", "Endocrinology Consultation: 35-year-old female presents with persistent fatigue, weight gain, cold intolerance, and constipation. Serum thyroid-stimulating hormone (TSH) markedly elevated at 14.2 mIU/L with decreased free thyroxine (FT4 0.6 ng/dL). Elevated anti-thyroid peroxidase antibodies confirm Hashimoto's autoimmune thyroiditis. Prescribed oral Levothyroxine 75 mcg daily on an empty stomach."),

        # FINANCIAL SAMPLES
        ("Financial", "Securities & Exchange Commission Form 10-K Annual Report: Consolidated Statements of Operations. Total net revenue for fiscal year ended December 31 reached $84.2 billion, representing an 18.5% year-over-year increase driven by cloud services and subscription monetization. Operating margin expanded 240 basis points to 31.2%. Diluted earnings per share (EPS) totaled $5.82 compared to $4.91 in the prior year. Free cash flow generation remained robust at $22.6 billion."),
        ("Financial", "Equity Research Analyst Report: Initiating coverage on tech sector enterprise software equities with an Outperform rating and $220 target price. The company exhibits strong net revenue retention (124%), expanding multi-product enterprise adoption, and a defensive gross margin profile of 78%. Key investment risks include macroeconomic cloud budget deceleration, foreign exchange headwinds, and increased capital expenditures on GPU computing clusters."),
        ("Financial", "Quarterly Earnings Call Transcript & Financial Guidance: During the third quarter, consolidated revenue reached $12.4 billion, exceeding consensus estimates by $350 million. Our core enterprise banking division delivered record net interest income amid the rising benchmark interest rate environment. For the fourth quarter, management guides revenue between $12.8B and $13.1B with expected non-GAAP operating margin of approximately 28.5%."),
        ("Financial", "Corporate Balance Sheet & Liquidity Analysis: Cash, cash equivalents, and marketable securities totaled $14.8 billion against total short-term commercial paper liabilities of $3.2 billion and long-term senior unsecured notes of $9.5 billion. Debt-to-equity ratio improved to 0.74x. Working capital metrics indicate strong solvency, with a current ratio of 2.1x and cash conversion cycle shortening by four days."),
        ("Financial", "Federal Reserve Monetary Policy & Interest Rate Statement: The Federal Open Market Committee decided to maintain the target range for the federal funds rate at 5.25% to 5.50%. Recent economic indicators suggest economic activity has been expanding at a solid pace. Inflation has eased over the past year but remains above the Committee's 2 percent objective. The Committee remains highly attentive to inflation risks and treasury yield dynamics."),
        ("Financial", "Mergers & Acquisitions Valuation Brief: Valuation analysis of the prospective asset buyout utilizing discounted cash flow (DCF) modeling and comparable company trading multiples. Assuming a weighted average cost of capital (WACC) of 9.2% and terminal growth rate of 2.5%, enterprise value is calculated between $4.2B and $4.6B, representing a 14.8x EV/EBITDA multiple."),
        ("Financial", "Hedge Fund Macro Portfolio Risk Assessment: Portfolio value-at-risk (VaR) measured at 99% confidence interval across multi-asset class exposures. Increased treasury curve steepening hedges via swaptions to mitigate duration risk. Credit default swap spreads in high-yield corporate debt widened by 35 basis points, prompting tactical reduction in leveraged loan allocations."),
        ("Financial", "Venture Capital Term Sheet & Series B Investment: Syndicate investors commit $35 million in Series B Preferred Stock financing at a pre-money valuation of $140 million. Participating preferred shares will carry an 8% cumulative dividend rate and 1x non-participating liquidation preference. Lead investor shall obtain one designated seat on the Board of Directors."),
        ("Financial", "Audited Financial Statements & Independent Auditor's Report: We have audited the consolidated balance sheets, statements of cash flows, and shareholder equity. In our professional opinion, the financial statements present fairly, in all material respects, the financial position of the corporation in conformity with generally accepted accounting principles (US GAAP)."),
        ("Financial", "Cryptocurrency Asset Class & Derivatives Market Report: Total digital asset market capitalization surpassed $2.4 trillion, supported by substantial institutional inflows into spot Bitcoin exchange-traded funds (ETFs). Open interest on CME futures reached historic highs, while options volatility smile reflects bullish call skew into calendar year-end expirations."),

        # TECHNOLOGY / ACADEMIC RESEARCH SAMPLES
        ("Technology & Academic", "Natural Language Processing Research Paper: We present a novel transformer architecture with linear-time multi-head self-attention mechanisms for long-context sequence modeling. By replacing standard quadratic attention with low-rank kernel approximations and flash attention caching, our model processes context windows up to 128,000 tokens with 4x higher inference throughput. Experiments on GLUE and SuperGLUE benchmarks show competitive downstream performance with 40% fewer parameters."),
        ("Technology & Academic", "Distributed Systems & Cloud Computing Conference: Designing a fault-tolerant, horizontally scalable key-value storage engine using the Raft consensus protocol. We implement multi-Raft partition groups across distributed datacenters, achieving sub-10 millisecond write latencies and automated leader election during network partitions. Evaluation under dynamic Byzantine failure scenarios demonstrates zero data loss and linearizability."),
        ("Technology & Academic", "Deep Reinforcement Learning for Robotic Manipulation: In this study, we propose an off-policy model-free actor-critic algorithm augmented with hindsight experience replay for multi-fingered robotic grasping. Simulation-to-reality transfer is achieved using domain randomization across physical friction parameters, visual textures, and sensor noise. Empirical benchmarks demonstrate a 94.6% task completion rate on real-world test objects."),
        ("Technology & Academic", "Computer Vision & Autonomous Perception: We introduce a hierarchical multi-scale feature pyramid network for real-time 3D bounding box detection from synchronized LiDAR point clouds and stereo camera streams. The multimodal sensor fusion pipeline utilizes cross-attention transformer layers to register sparse spatial coordinates, outperforming prior state-of-the-art methods by 3.8% mAP on the KITTI benchmark."),
        ("Technology & Academic", "Quantum Computing & Quantum Information Theory: We demonstrate fault-tolerant quantum error correction on a superconducting transmon processor implementing a surface code with distance five. Using cryogenic microwave control lines and real-time FPGA syndrome decoding, we demonstrate physical logical qubit lifetime exceeding the break-even threshold compared to constituent unencoded physical qubits."),
        ("Technology & Academic", "Cybersecurity & Microarchitectural Vulnerabilities: Analysis of speculative execution side-channel attacks targeting transient instruction pipelines in modern out-of-order processors. We demonstrate that branch predictor buffer aliasing enables cache timing leakages that extract AES encryption keys from isolated virtual machine enclaves. We propose architectural barrier mitigations and software mitigations."),
        ("Technology & Academic", "Bioinformatics & Genomic Sequence Modeling: High-throughput sequence alignment using deep bidirectional autoregressive models. We train self-supervised masked nucleotide models on 50 million metagenomic contigs to predict non-coding regulatory variants and transcription factor binding motifs. Validation against ChIP-seq assays confirms significant predictive gains over traditional position weight matrices."),
        ("Technology & Academic", "Database Engineering & Query Optimization: Learned query optimization for distributed analytical database engines. We replace traditional cost-based dynamic programming query planners with tree-convolutional neural networks that estimate cardinality directly from raw database table metadata and historical query execution logs. Empirical evaluation on TPC-DS benchmarks shows 32% faster query completion."),
        ("Technology & Academic", "Edge Computing & Neural Model Quantization: We investigate post-training 4-bit integer quantization and structured pruning for large language models deployed on edge embedded processors. By applying optimal brain compression and second-order Taylor expansion weighting, we compress 7-billion parameter checkpoints to 3.8 GB while preserving 99.2% of baseline perplexity."),
        ("Technology & Academic", "Graph Neural Networks for Molecule Discovery: Predicting molecular solubility and binding affinity using message-passing neural networks over chemical molecular graph representations. Atoms are parameterized as graph nodes with covalent bonds as directed edges. Our model achieves state-of-the-art mean squared error on the MoleculeNet benchmark and successfully prioritizes novel therapeutic candidate compounds.")
    ]

    # Expand corpus programmatically with diverse variations to reach robust training size (240 rich documents)
    full_data = []
    domains = ["Legal", "Healthcare", "Financial", "Technology & Academic"]
    domain_keywords = {
        "Legal": ["jurisdiction", "statutory", "injunction", "arbitration", "tort", "covenant", "plaintiff", "deposition", "affidavit", "intellectual property", "severance", "compliance", "settlement", "indemnity", "breach of contract"],
        "Healthcare": ["diagnosis", "pathology", "symptoms", "pharmacology", "clinical trial", "oncology", "patient", "biopsy", "intravenous", "syndrome", "radiology", "therapy", "surgical", "prognosis", "vital signs"],
        "Financial": ["revenue", "earnings per share", "balance sheet", "liquidity", "dividend", "equity", "cash flow", "marketable securities", "operating margin", "valuation", "EBITDA", "fiscal year", "interest rate", "portfolio", "treasury"],
        "Technology & Academic": ["neural network", "transformer", "latency", "benchmark", "quantization", "algorithm", "hyperparameter", "distributed systems", "dataset", "accuracy", "point cloud", "throughput", "parameter", "loss function", "architecture"]
    }

    for dom, text in docs:
        full_data.append({"domain": dom, "text": text})
        
    # Generate variations with authentic domain context
    for i in range(5):
        for dom, base_text in docs:
            kw = domain_keywords[dom][i % len(domain_keywords[dom])]
            var_text = f"Supplementary {dom} Document {i+1}: In regarding to {kw} and associated proceedings, " + base_text + f" Key considerations regarding {kw} were systematically addressed during the review cycle."
            full_data.append({"domain": dom, "text": var_text})
            
    df = pd.DataFrame(full_data)
    df.to_csv(csv_path, index=False)
    print(f"Generated domain corpus with {len(df)} documents across {len(df['domain'].unique())} domains.")
    return df

def train_and_evaluate_domain_classifier():
    """Train, evaluate, and save domain classifier models."""
    df = create_domain_corpus()
    
    # Preprocess texts
    df['cleaned_text'] = df['text'].apply(preprocess_domain_text)
    
    X = df['cleaned_text']
    y = df['domain']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train candidate models
    models = {
        "Calibrated Logistic Regression": CalibratedClassifierCV(LogisticRegression(max_iter=1000, C=2.0)),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Passive-Aggressive (Calibrated)": CalibratedClassifierCV(PassiveAggressiveClassifier(max_iter=1000, random_state=42))
    }
    
    benchmark_results = {}
    best_name = None
    best_f1 = -1
    best_model = None
    
    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        
        benchmark_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "macro_f1": round(float(f1), 4)
        }
        print(f"[{name}] Acc: {acc:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = clf
            
    # Save best model and vectorizer
    model_artifact = {
        "model": best_model,
        "vectorizer": vectorizer,
        "classes": list(best_model.classes_),
        "best_name": best_name,
        "benchmark": benchmark_results
    }
    
    model_save_path = os.path.join(MODELS_DIR, "domain_classifier.joblib")
    joblib.dump(model_artifact, model_save_path)
    print(f"Saved best domain classifier ({best_name}) to {model_save_path}")
    
    # Generate Confusion Matrix
    y_pred_best = best_model.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=best_model.classes_,
                yticklabels=best_model.classes_)
    plt.title(f"Project 15: Domain Classifier Confusion Matrix\n({best_name})", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Domain", fontsize=10, fontweight='bold')
    plt.ylabel("Actual Domain", fontsize=10, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(VIS_DIR, "confusion_matrix_domain.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved domain confusion matrix to {cm_path}")
    
    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "domain_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
        
    return benchmark_results

class DomainInferenceEngine:
    """Inference and summarization engine for Project 15."""
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "domain_classifier.joblib")
        if not os.path.exists(model_path):
            print("Domain model not found, training now...")
            train_and_evaluate_domain_classifier()
        data = joblib.load(model_path)
        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        self.classes = data["classes"]
        self.best_name = data["best_name"]
        
    def classify_document(self, raw_text: str):
        """Classify document text, compute confidence scores, and extract summary."""
        if not raw_text or len(raw_text.strip()) == 0:
            return {
                "error": "Document input is empty. Please paste text or upload a document."
            }
            
        cleaned = preprocess_domain_text(raw_text)
        if len(cleaned.strip()) == 0:
            return {
                "error": "Document contains no readable alphabetic content after preprocessing."
            }
            
        vec = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(vec)[0]
        pred_idx = np.argmax(probs)
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.classes, probs)}
        
        # Summary
        summary = extractive_summarize(raw_text, max_sentences=3)
        
        # Key domain keywords detected
        feature_names = self.vectorizer.get_feature_names_out()
        nonzero_indices = vec.nonzero()[1]
        token_scores = [(feature_names[i], vec[0, i]) for i in nonzero_indices]
        token_scores.sort(key=lambda x: x[1], reverse=True)
        top_cues = [word for word, score in token_scores[:6]]
        
        return {
            "predicted_domain": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "summary": summary,
            "key_cues": top_cues,
            "word_count": len(raw_text.split()),
            "character_count": len(raw_text)
        }

if __name__ == "__main__":
    train_and_evaluate_domain_classifier()
