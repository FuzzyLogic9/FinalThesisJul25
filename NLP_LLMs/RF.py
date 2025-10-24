import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
from sklearnex import patch_sklearn

# ✅ Intel optimization
patch_sklearn()

# 📌 Paths
input_file = "C:/Users/fzkar/final_code_data/matched_records_70K_toolbox_all_final_sentiment_scores_wo_text.csv"
model_out_path = "C:/Users/fzkar/final_code_data/rf_model_gridsearch.pkl"
scaler_out_path = "C:/Users/fzkar/final_code_data/scaler_rf_gridsearch.pkl"
metrics_txt_path = "C:/Users/fzkar/final_code_data/rf_metrics_report.txt"
feature_importance_path = "C:/Users/fzkar/final_code_data/rf_feature_importance.csv"
summary_csv_path = "C:/Users/fzkar/final_code_data/rf_summary_metrics.csv"

# 📌 Load data
df = pd.read_csv(input_file, encoding='utf-8')
features = [
    "Stanza_Sentiment", "VADER_Negative", "VADER_Neutral", "VADER_Positive",
    "VADER_Compound", "TextBlob_Polarity", "Pattern_Polarity", "HB_toxicity_score"
]
target = "Multiclass Average"

X = df[features]
y = df[target]

# 📌 Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ SMOTE for class balance
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 📌 Random Forest + GridSearch
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_bal, y_train_bal)

# 📌 Best model evaluation
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
conf_matrix = confusion_matrix(y_test, y_pred)

# ✅ Macro-average metrics
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro'
)

# ✅ Save metrics as text
with open(metrics_txt_path, "w") as f:
    f.write("Random Forest Classifier Report\n")
    f.write("="*50 + "\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix, separator=', ') + "\n")

# ✅ Save summary metrics to CSV
summary_metrics = {
    "Accuracy": [acc],
    "Macro Precision": [precision_macro],
    "Macro Recall": [recall_macro],
    "Macro F1": [f1_macro]
}
pd.DataFrame(summary_metrics).to_csv(summary_csv_path, index=False)

# ✅ Save feature importances
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": best_rf.feature_importances_
})
importance_df.sort_values(by="Importance", ascending=False).to_csv(feature_importance_path, index=False)

# ✅ Save model and scaler
joblib.dump(best_rf, model_out_path)
joblib.dump(scaler, scaler_out_path)

print("✅ Model training complete. All metrics and artifacts saved.")
