import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, roc_auc_score
)
import warnings
import os
import time

warnings.filterwarnings('ignore')

# ✅ Use only one SVC model
MODELS = {
    "SVC_Linear": SVC(kernel="linear", probability=True)
}

def evaluate_model(model, X, y, model_name, class_name):
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    metrics = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = make_pipeline(StandardScaler(), model)
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

        metrics.append({
            "TrainTimeSec": end - start,
            "Model": model_name,
            "Class": class_name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average='macro', zero_division=0),
            "Recall": recall_score(y_test, preds, average='macro', zero_division=0),
            "F1": f1_score(y_test, preds, average='macro', zero_division=0),
            "CohenKappa": cohen_kappa_score(y_test, preds),
            #"ROC_AUC": roc_auc_score(y_test, probs, multi_class='ovo', average='macro')
             "ROC_AUC": roc_auc_score(y_test, probs[:, 1]) if probs.shape[1] == 2 else roc_auc_score(y_test, probs, multi_class='ovo', average='macro')

        })
    return metrics

def run_all_models(input_file):
    df = pd.read_csv(input_file)
    output_dir = "C:/Users/fzkar/final_code_data/SVC_outputs"
    os.makedirs(output_dir, exist_ok=True)

    X = df.iloc[:, :8].values  # adjust if necessary

    class_labels = {
        "Aggression AVG": "Humannotated Aggression Class Average Wholed",
        "Aggression OneInTen": "Humannotated Aggression Class",
        "Attacks AVG": "Humannotated Attack Class Average Wholed",
        "Attacks OneInTen": "Humannotated Attack Class",
        "Toxic AVG": "Humannotated Toxic Class Average Wholed",
        "Toxic OneInTen": "Humannotated Toxic Class",
        "Multiclass AVG": "Multiclass Average",
        "Multiclass OneInTen": "MultiClass"
    }

    results = []
    for trait_label, col_name in class_labels.items():
        y = df[col_name].values.astype(int)
        for model_name, model in MODELS.items():
            results.extend(evaluate_model(model, X, y, model_name, trait_label))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "model_evaluation_results.csv"), index=False)
    print("Saved model evaluation results.")

    for metric in ["Accuracy", "F1", "Precision", "Recall", "ROC_AUC"]:
        pivot = results_df.pivot_table(index="Model", columns="Class", values=metric)
        pivot.to_csv(os.path.join(output_dir, f"combined_{metric.lower()}_table.csv"))
        print(f"Saved combined {metric} table.")

        plt.figure(figsize=(14, 8))
        cmap = "coolwarm" if metric != "ROC_AUC" else "viridis"
        sns.heatmap(pivot, annot=True, cmap=cmap, fmt=".2f")
        plt.title(f"{metric} of Classifier per Trait")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric.lower()}_heatmap.png"))
        plt.close()

        for trait in pivot.columns:
            plt.figure(figsize=(10, 6))
            pivot[trait].sort_values(ascending=False).plot(kind='bar')
            plt.title(f"{metric} Scores for '{trait}'")
            plt.ylabel(metric)
            plt.xlabel("Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric.lower()}_bar_{trait.replace(' ', '_')}.png"))
            plt.close()

if __name__ == "__main__":
    run_all_models("C:/Users/fzkar/final_code_data/matched_records_70K_toolbox_all_final_sentiment_scores_wo_text.csv")
