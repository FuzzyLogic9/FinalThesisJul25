import pandas as pd
import matplotlib.pyplot as plt

# === STEP 1: INPUT FILES AND COLUMNS ===
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

# Columns to include in the toxicity average
score_columns = [
    "HB_toxicity_score",
   "VADER_Negative",
    "VADER_Compound",
    "TextBlob_Polarity",
    "Pattern_Polarity",
    "Stanza_Sentiment"
]

# Columns where lower = worse → flip them before averaging
flip_columns = [
    "VADER_Compound",
    "TextBlob_Polarity",
    "Pattern_Polarity",
    "Stanza_Sentiment"
]

# === STEP 2: CALCULATE FLIPPED AVERAGES PER GROUP ===
group_averages = {}

for group, file in files.items():
    df = pd.read_csv(file)
    df_filtered = df[score_columns].copy()

    # Flip necessary columns so higher = worse for all
    for col in flip_columns:
        if col in df_filtered.columns:
            df_filtered[col] = -df_filtered[col]

    # Optional: print the means after flipping
    print(f"\n🔍 {group} column means after flipping:")
    print(df_filtered.mean())

    # Compute simple average across the 6 columns
    simple_avg = df_filtered.mean().mean()
    print(f"✅ {group} simple average toxicity score: {simple_avg:.6f}")

    group_averages[group] = simple_avg

# === STEP 3: CREATE SUMMARY DATAFRAME ===
result_df = pd.DataFrame.from_dict(group_averages, orient='index', columns=["Simple_Average_Toxicity"])

# Save CSV
output_csv_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/simple_average_toxicity_summary.csv"
result_df.to_csv(output_csv_path)
print(f"\n[💾] Saved average summary to: {output_csv_path}")

# === STEP 4: PLOT BAR CHART ===
plt.figure(figsize=(8, 5))
result_df["Simple_Average_Toxicity"].plot(kind='bar', color=['crimson', 'tomato'])
plt.title("Simple Average Toxicity Score (All Flipped to Higher = Worse)")
plt.ylabel("Average Toxicity (Higher = More Toxic)")
plt.xticks(rotation=0)
plt.tight_layout()

output_img_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/simple_average_toxicity_plot.png"
plt.savefig(output_img_path)
plt.show()
print(f"[🖼️] Plot saved to: {output_img_path}")

# === STEP 5: IDENTIFY MOST TOXIC GROUP ===
most_toxic_group = result_df["Simple_Average_Toxicity"].idxmax()
most_toxic_score = result_df["Simple_Average_Toxicity"].max()

print(f"\n🚨 Most toxic group (simple average): **{most_toxic_group}** with score: {most_toxic_score:.6f}")

# Save plain text result
with open("C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/most_toxic_simple_average.txt", "w") as f:
    f.write(f"Most toxic group (simple average): {most_toxic_group}\nScore: {most_toxic_score:.6f}")
