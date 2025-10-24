import pandas as pd
import matplotlib.pyplot as plt

# === STEP 1: FILES AND COLUMN ===
files = {
    "GPTNEO_0": "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPTNEO_G0_OT_scores.csv",
    "GPTNEO_1": "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPTNEO_G1_OT_scores.csv",
    "GPT2_0": "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPT2_G0_OT_scores.csv",
    "GPT2_1": "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPT2_G1_OT_scores.csv",
    "GPTNEO_FT_0": "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPTNEO_FT_G0_OT_scores.csv",
    "GPTNEO_FT_1": "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPTNEO_FT_G1_OT_scores.csv",
    "GPT2_FT_0": "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPT2_FT_G0_OT_scores.csv",
    "GPT2_FT_1": "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPT2_FT_G1_OT_scores.csv",
    "Original": "C:/Users/fzkar/final_code_data/matched_records_70K_toolbox_all_final_sentiment_scores_wo_text.csv"
}

score_column = "HB_toxicity_score"

# === STEP 2: CALCULATE AVERAGE HATEBERT SCORE PER GROUP ===
group_averages = {}

for group, file in files.items():
    df = pd.read_csv(file)

    if score_column not in df.columns:
        raise ValueError(f"{score_column} not found in {file}")
    
    mean_score = df[score_column].mean()
    group_averages[group] = mean_score

    print(f"✅ {group} HateBERT average toxicity score: {mean_score:.6f}")

# === STEP 3: CREATE RESULT TABLE ===
result_df = pd.DataFrame.from_dict(group_averages, orient='index', columns=["HateBERT_Average_Toxicity"])

# Save CSV
output_csv_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/hatebert_average_summary.csv"
result_df.to_csv(output_csv_path)
print(f"\n[💾] Saved HateBERT summary to: {output_csv_path}")

# === STEP 4: PLOT ===
plt.figure(figsize=(12, 5))
result_df["HateBERT_Average_Toxicity"].plot(kind='bar', color=['darkred', 'orangered'])
plt.title("HateBERT Average Toxicity Score by Group")
plt.ylabel("Average Toxicity (Higher = More Toxic)")
plt.xticks(rotation=0)
plt.tight_layout()

output_img_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/hatebert_average_plot.png"
plt.savefig(output_img_path)
plt.show()
print(f"[🖼️] Plot saved to: {output_img_path}")

# === STEP 5: IDENTIFY MOST TOXIC GROUP ===
most_toxic_group = result_df["HateBERT_Average_Toxicity"].idxmax()
most_toxic_score = result_df["HateBERT_Average_Toxicity"].max()

print(f"\n🚨 Most toxic group (HateBERT only): **{most_toxic_group}** with score: {most_toxic_score:.6f}")

# Save plain text summary
with open("C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/most_toxic_hatebert.txt", "w") as f:
    f.write(f"Most toxic group (HateBERT only): {most_toxic_group}\nScore: {most_toxic_score:.6f}")
