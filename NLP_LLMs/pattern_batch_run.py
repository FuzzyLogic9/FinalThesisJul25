import pandas as pd
from pattern3.en import sentiment as pattern_sentiment

# Load CSV
df = pd.read_csv("C:/Users/fzkar/final_code_data/pattern_snap.csv")  # <-- Change this path as needed

# Check column name
text_column = "text"
df[text_column] = df[text_column].astype(str)

# Apply Pattern Sentiment
pattern_scores = df[text_column].apply(lambda x: pattern_sentiment(x))

# Split into polarity and subjectivity
df['Pattern_Polarity'] = pattern_scores.apply(lambda x: x[0])
#df['Pattern_Subjectivity'] = pattern_scores.apply(lambda x: x[1])

# Save to new CSV
df.to_csv("C:/Users/fzkar/final_code_data/pattern_snap_out2.csv", index=False)
print("✅ Sentiment analysis complete. Output saved as 'pattern_snap_out2.csv'")
