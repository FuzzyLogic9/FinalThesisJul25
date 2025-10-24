import pandas as pd
import joblib
import os

# === 📍 Paths ===
MODEL_PATH = "C:/Users/fzkar/final_code_data/rf_model_gridsearch.pkl"
SCALER_PATH = "C:/Users/fzkar/final_code_data/scaler_rf_gridsearch.pkl"
NEW_DATA_PATH = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPTNEO_FT_G1_OT_scores.csv"
OUTPUT_PATH = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPTNEO_FT_G1_OT_predicted.csv"
#TOP5_OUTPUT_PATH = "C:/Users/fzkar/final_code_data/top5_predictions.csv"

# === 🔢 Features used in training ===
FEATURE_COLUMNS = [
    "Stanza_Sentiment", "VADER_Negative", "VADER_Neutral", "VADER_Positive",
    "VADER_Compound", "TextBlob_Polarity", "Pattern_Polarity", "HB_toxicity_score"
]

# === 🛂 Page ID Column (change if named differently)
PAGE_ID_COL = "PageID"  # 🔄 Change to actual column name if needed

# === ✅ Load model and scaler
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or Scaler file not found.")

rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === 📄 Load new data
new_df = pd.read_csv(NEW_DATA_PATH)

# ✅ Validate presence of required columns
missing_cols = [col for col in FEATURE_COLUMNS if col not in new_df.columns]
if PAGE_ID_COL not in new_df.columns:
    raise ValueError(f"Missing required column: {PAGE_ID_COL}")
if missing_cols:
    raise ValueError(f"Missing required feature columns: {missing_cols}")

# === 🔍 Scale feature values
X_new = new_df[FEATURE_COLUMNS]
X_new_scaled = scaler.transform(X_new)

# === 🧠 Predict
predictions = rf_model.predict(X_new_scaled)
probs = rf_model.predict_proba(X_new_scaled)
class_labels = rf_model.classes_

# === 📝 Build output DataFrame
output_df = new_df.copy()
output_df["Predicted Class"] = predictions
#output_df["Prediction Confidence"] = probs.max(axis=1)

# === Add softmax-like probabilities for all classes
#for i, label in enumerate(class_labels):
#   output_df[f"Prob_Class_{label}"] = probs[:, i]

# === Save full prediction results
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Full predictions saved to: {OUTPUT_PATH}")

# === 📊 Extract top 5 most confident predictions
#top5 = output_df[[PAGE_ID_COL, "Predicted Class", "Prediction Confidence"]]
#top5_sorted = top5.sort_values(by="Prediction Confidence", ascending=False).head(5)
#top5_sorted.to_csv(TOP5_OUTPUT_PATH, index=False)
#print(f"✅ Top 5 confident predictions saved to: {TOP5_OUTPUT_PATH}")
