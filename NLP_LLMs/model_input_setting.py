import pandas as pd

# === Load the data ===
df = pd.read_csv("C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPTNEO_FT_G0_OT_scores.csv")

# === Reorder columns ===
# Move stanza_sentiment to the front, hatebert_score to the end
# === Rename page_id to Page_ID (for model compatibility)
df.rename(columns={'page_id': 'PageID'}, inplace=True)
df_reordered = df.copy()
first = df.pop("stanza_sentiment")
last = df.pop("hatebert_score")
df.insert(0, "stanza_sentiment", first)
df["hatebert_score"] = last

# === Rename columns to match model expectations ===
df_renamed = df.rename(columns={
    'stanza_sentiment': 'Stanza_Sentiment',
    'neg': 'VADER_Negative',
    'neu': 'VADER_Neutral',
    'pos': 'VADER_Positive',
    'compound': 'VADER_Compound',
    'polarity': 'TextBlob_Polarity',
    'pattern_polarity': 'Pattern_Polarity',
    'hatebert_score': 'HB_toxicity_score'
})

# === Rearrange: page_id + model features
model_input_df = df_renamed[[
    "PageID", "level", # retain for output tracking
    "Stanza_Sentiment", "VADER_Negative", "VADER_Neutral", "VADER_Positive",
    "VADER_Compound", "TextBlob_Polarity", "Pattern_Polarity", "HB_toxicity_score"
]]

# === Save output ===
model_input_df.to_csv("C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Final_GPTNEO_FT_G0_OT_scores.csv", index=False)
print("✅ Done! File for model input is ready.'")
