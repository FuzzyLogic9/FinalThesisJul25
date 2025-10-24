import pandas as pd

# === File Paths ===
score_file_path = 'C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/score_output_escalated_GPTNEO_FT_G1_OT.csv'  # Replace with actual path
review_file_path = 'C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/text_output_escalated_GPTNEO_FT_G1_OT.csv'  # Replace with actual path

# === Load Data ===
df_scores = pd.read_csv(score_file_path)
df_reviews = pd.read_csv(review_file_path)

# === Simulate Pattern Polarity (you should replace this with actual pattern polarity values) ===
# For now, let's mock with random polarity values (or integrate Pattern library if needed)
from pattern3.en import sentiment as pattern_sentiment

# === Compute Pattern Polarity from 'toxic' column ===
df_reviews['pattern_polarity'] = df_reviews['toxic'].apply(
    lambda x: pattern_sentiment(str(x))[0] if pd.notna(x) else 0.0
)

# Keep only what's needed for the merge
df_reviews = df_reviews[['page_id', 'level', 'pattern_polarity']]

# Drop any old subjectivity column just in case
df_scores = df_scores.drop(columns=['subjectivity'], errors='ignore')

# Merge new pattern_polarity
df_scores = pd.merge(df_scores, df_reviews, on=['page_id', 'level'], how='left')

# === Reorder: move pattern_polarity before stanza_sentiment ===
cols = df_scores.columns.tolist()
if 'pattern_polarity' in cols and 'stanza_sentiment' in cols:
    cols.remove('pattern_polarity')
    insert_at = cols.index('stanza_sentiment')
    cols = cols[:insert_at] + ['pattern_polarity'] + cols[insert_at:]

df_scores = df_scores[cols]

# === Save Updated Scores ===
df_scores.to_csv('C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPTNEO_FT_G1_OT_scores.csv', index=False)
print("✅ Scores file updated with pattern polarity.")
