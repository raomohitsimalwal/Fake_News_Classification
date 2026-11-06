"""
Model Training, Feature Engineering, and Evaluation Pipeline.

Implements:
1. TF-IDF feature extraction with n-gram range (1, 2) and sublinear term frequency scaling.
2. Training and comparison of multiple classification algorithms:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear Support Vector Machine (LinearSVC with CalibratedClassifierCV)
   - Passive Aggressive Classifier
   - Random Forest Classifier
3. Comprehensive evaluation: Accuracy, Precision, Recall, F1-score (Macro and Weighted), Confusion Matrix.
4. Model selection based on validation/test performance.
5. Model persistence (Joblib) for the Streamlit web GUI.
6. Generation of confusion matrix and model comparison plots.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")


def train_and_evaluate_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    print("=" * 70)
    print("NLP CAPSTONE: MODEL DEVELOPMENT, EVALUATION & FEATURE ENGINEERING")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # PART 1: News Article Classification (Real vs Fake)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. TRAINING NEWS ARTICLE CLASSIFIER (Real vs Fake)")
    print("=" * 50)

    df_news = pd.read_csv(os.path.join(DATA_DIR, "news_processed.csv"))
    X_news = df_news["clean_full_text"].fillna("")
    y_news = df_news["label"]

    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
        X_news, y_news, test_size=0.20, random_state=42, stratify=y_news
    )
    print(f"News Dataset - Train: {len(X_train_n)}, Test: {len(X_test_n)}")

    # TF-IDF Feature Engineering
    news_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode"
    )
    X_train_n_vec = news_vectorizer.fit_transform(X_train_n)
    X_test_n_vec = news_vectorizer.transform(X_test_n)

    # Models to compare for News
    news_candidates = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0, random_state=42),
        "Passive Aggressive (Calibrated)": CalibratedClassifierCV(PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42)),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
    }

    news_results = {}
    best_news_acc = 0.0
    best_news_model = None
    best_news_name = ""

    for name, clf in news_candidates.items():
        clf.fit(X_train_n_vec, y_train_n)
        preds = clf.predict(X_test_n_vec)

        acc = accuracy_score(y_test_n, preds)
        prec_macro = precision_score(y_test_n, preds, average="macro")
        rec_macro = recall_score(y_test_n, preds, average="macro")
        f1_macro = f1_score(y_test_n, preds, average="macro")
        prec_wt = precision_score(y_test_n, preds, average="weighted")
        rec_wt = recall_score(y_test_n, preds, average="weighted")
        f1_wt = f1_score(y_test_n, preds, average="weighted")

        news_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_weighted": round(float(prec_wt), 4),
            "recall_weighted": round(float(rec_wt), 4),
            "f1_weighted": round(float(f1_wt), 4)
        }
        print(f"  -> {name:<26} | Acc: {acc:.4f} | F1 (Macro): {f1_macro:.4f} | Prec: {prec_macro:.4f} | Rec: {rec_macro:.4f}")

        if acc > best_news_acc:
            best_news_acc = acc
            best_news_model = clf
            best_news_name = name

    print(f"\n>> Best News Classifier: {best_news_name} with Accuracy = {best_news_acc * 100:.2f}%")

    # Evaluate best news model confusion matrix
    best_news_preds = best_news_model.predict(X_test_n_vec)
    cm_news = confusion_matrix(y_test_n, best_news_preds, labels=["REAL", "FAKE"])
    report_news = classification_report(y_test_n, best_news_preds, output_dict=True)

    # Save News Models
    joblib.dump(best_news_model, os.path.join(MODEL_DIR, "news_classifier.joblib"))
    joblib.dump(news_vectorizer, os.path.join(MODEL_DIR, "news_vectorizer.joblib"))
    print("Saved best news model & vectorizer to models/")

    # -------------------------------------------------------------------------
    # PART 2: Claim Credibility Classification (LIAR Benchmark)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. TRAINING CLAIM CREDIBILITY CLASSIFIER (LIAR Benchmark)")
    print("=" * 50)

    df_liar = pd.read_csv(os.path.join(DATA_DIR, "liar_processed.csv"))
    # Use clean_statement + context for rich claim modeling
    X_claim = df_liar["clean_rich_text"].fillna("")
    y_claim_6 = df_liar["label"]
    y_claim_3 = df_liar["credibility_3class"]

    # 3-Tier Credibility (Credible, Half-True, Questionable)
    X_train_c, X_test_c, y_train_c3, y_test_c3 = train_test_split(
        X_claim, y_claim_3, test_size=0.20, random_state=42, stratify=y_claim_3
    )
    print(f"Claim Dataset - Train: {len(X_train_c)}, Test: {len(X_test_c)}")

    # TF-IDF for Claims
    claim_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode"
    )
    X_train_c_vec = claim_vectorizer.fit_transform(X_train_c)
    X_test_c_vec = claim_vectorizer.transform(X_test_c)

    claim_candidates = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.5, class_weight="balanced", random_state=42),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=42)),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, C=0.3, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1)
    }

    claim_results = {}
    best_claim_f1 = 0.0
    best_claim_model = None
    best_claim_name = ""

    for name, clf in claim_candidates.items():
        clf.fit(X_train_c_vec, y_train_c3)
        preds = clf.predict(X_test_c_vec)

        acc = accuracy_score(y_test_c3, preds)
        prec_macro = precision_score(y_test_c3, preds, average="macro")
        rec_macro = recall_score(y_test_c3, preds, average="macro")
        f1_macro = f1_score(y_test_c3, preds, average="macro")
        prec_wt = precision_score(y_test_c3, preds, average="weighted")
        rec_wt = recall_score(y_test_c3, preds, average="weighted")
        f1_wt = f1_score(y_test_c3, preds, average="weighted")

        claim_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_weighted": round(float(prec_wt), 4),
            "recall_weighted": round(float(rec_wt), 4),
            "f1_weighted": round(float(f1_wt), 4)
        }
        print(f"  -> {name:<26} | Acc: {acc:.4f} | F1 (Macro): {f1_macro:.4f} | Prec: {prec_macro:.4f} | Rec: {rec_macro:.4f}")

        # In multi-class imbalanced claim benchmarks, macro F1 is the academic gold standard
        if f1_macro > best_claim_f1:
            best_claim_f1 = f1_macro
            best_claim_model = clf
            best_claim_name = name

    print(f"\n>> Best Claim Classifier: {best_claim_name} with Macro F1 = {best_claim_f1:.4f} (Accuracy = {claim_results[best_claim_name]['accuracy']*100:.2f}%)")

    # Evaluate best claim model confusion matrix
    best_claim_preds = best_claim_model.predict(X_test_c_vec)
    claim_labels_order = ["Credible", "Half-True", "Questionable"]
    cm_claim = confusion_matrix(y_test_c3, best_claim_preds, labels=claim_labels_order)
    report_claim = classification_report(y_test_c3, best_claim_preds, output_dict=True)

    # Save Claim Models
    joblib.dump(best_claim_model, os.path.join(MODEL_DIR, "claim_classifier.joblib"))
    joblib.dump(claim_vectorizer, os.path.join(MODEL_DIR, "claim_vectorizer.joblib"))
    print("Saved best claim model & vectorizer to models/")

    # -------------------------------------------------------------------------
    # PART 3: Generate Evaluation Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. GENERATING CONFUSION MATRICES & EVALUATION CHARTS")
    print("=" * 50)

    # Plot News Confusion Matrix
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm_news, annot=True, fmt="d", cmap="Blues",
        xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"],
        cbar=True, annot_kws={"size": 13, "fontweight": "bold"}
    )
    plt.title(f"Confusion Matrix: News Articles ({best_news_name})\nAccuracy: {best_news_acc*100:.2f}%", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.tight_layout()
    cm_news_path = os.path.join(VIS_DIR, "confusion_matrix_news.png")
    plt.savefig(cm_news_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_news_path}")

    # Plot Claims Confusion Matrix
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm_claim, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=claim_labels_order, yticklabels=claim_labels_order,
        cbar=True, annot_kws={"size": 13, "fontweight": "bold"}
    )
    plt.title(f"Confusion Matrix: Fact-Check Claims ({best_claim_name})\nAccuracy: {claim_results[best_claim_name]['accuracy']*100:.2f}%", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Credibility Tier", fontsize=11)
    plt.ylabel("Actual Credibility Tier", fontsize=11)
    plt.tight_layout()
    cm_claim_path = os.path.join(VIS_DIR, "confusion_matrix_claims.png")
    plt.savefig(cm_claim_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_claim_path}")

    # Plot Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # News comparison
    news_df_plot = pd.DataFrame(news_results).T[["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
    news_df_plot.columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    news_df_plot.plot(kind="bar", ax=axes[0], colormap="viridis", edgecolor="#333333", width=0.75)
    axes[0].set_title("News Classifier Algorithms Benchmark", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0.7, 1.0)
    axes[0].legend(loc="lower right")

    # Claim comparison
    claim_df_plot = pd.DataFrame(claim_results).T[["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
    claim_df_plot.columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    claim_df_plot.plot(kind="bar", ax=axes[1], colormap="magma", edgecolor="#333333", width=0.75)
    axes[1].set_title("Claim Credibility Algorithms Benchmark (LIAR)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(0.3, 0.8)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    comp_plot_path = os.path.join(VIS_DIR, "model_comparison_chart.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_plot_path}")

    # -------------------------------------------------------------------------
    # PART 4: Export Evaluation Metrics to JSON
    # -------------------------------------------------------------------------
    eval_metrics = {
        "best_news_model_name": best_news_name,
        "best_news_accuracy": round(float(best_news_acc), 4),
        "news_models_comparison": news_results,
        "news_classification_report": report_news,
        "news_confusion_matrix": cm_news.tolist(),
        "best_claim_model_name": best_claim_name,
        "best_claim_macro_f1": round(float(best_claim_f1), 4),
        "best_claim_accuracy": claim_results[best_claim_name]["accuracy"],
        "claim_models_comparison": claim_results,
        "claim_classification_report": report_claim,
        "claim_confusion_matrix": cm_claim.tolist(),
        "claim_categories": claim_labels_order
    }

    metrics_path = os.path.join(MODEL_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=4)
    print(f"Exported complete evaluation metrics: {metrics_path}")

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED, EVALUATED, AND SAVED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate_models()
