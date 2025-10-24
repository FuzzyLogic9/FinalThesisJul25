import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import stanza
from tqdm import tqdm
from datetime import datetime

# ==================== SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# GPT-2 Model (Generation)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model.eval()
torch.cuda.empty_cache()  # ✅ Clear GPU memory after model loading

# HateBERT (Toxicity Detection)
hatebert_path = "C:/FuzzyNov2024/hatebert_finetuned"
hatebert_tokenizer = AutoTokenizer.from_pretrained(hatebert_path)
hatebert_model = AutoModelForSequenceClassification.from_pretrained(hatebert_path).to(device)
hatebert_model.eval()
torch.cuda.empty_cache()  # ✅ Clear GPU memory after model loading

# VADER
vader = SentimentIntensityAnalyzer()

# StanfordNLP
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,sentiment', use_gpu=torch.cuda.is_available())

# ==================== FUNCTIONS ====================
def get_hatebert_score(text):
    try:
        inputs = hatebert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = hatebert_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            return scores[0][1].item()  # Probability of toxic
    except:
        return 0.0

def get_vader_score(text):
    return vader.polarity_scores(text)

def get_textblob_score(text):
    tb = TextBlob(text)
    return {
        "polarity": tb.polarity,
        "subjectivity": tb.subjectivity
    }

def get_stanford_score(text):
    try:
        doc = nlp(text)
        return doc.sentences[0].sentiment if doc.sentences else 1
    except:
        return 1

def generate_text(prompt, temperature=1.0, top_k=50, top_p=0.95, max_new_tokens=10):
    inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    input_len = inputs.input_ids.shape[1]

    if input_len + max_new_tokens > gpt_model.config.n_positions:
        inputs.input_ids = inputs.input_ids[:, -gpt_model.config.n_positions + max_new_tokens:]

    with torch.no_grad():
        output = gpt_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=gpt_tokenizer.pad_token_id
        )
    return gpt_tokenizer.decode(output[0], skip_special_tokens=True)

def score_text(row):
    scores = {
        "hatebert_score": get_hatebert_score(row['text']),
        **get_vader_score(row['text']),
        **get_textblob_score(row['text']),
        "stanford_sentiment": get_stanford_score(row['text'])
    }
    return pd.Series(scores)

# ==================== LOAD DATA ====================
base_path = "C:/FuzzyNov2024/toxicity_project/final_code_data/"
reviews_file = base_path + "group_zero_reviews.csv"
target_levels_file = base_path + "target_levels.csv"

text_only_toxic_file = base_path + "text_output_escalated.csv"
text_only_counter_file = base_path + "text_output_deescalated.csv"
scores_only_toxic_file = base_path + "score_output_escalated.csv"
scores_only_counter_file = base_path + "score_output_deescalated.csv"

reviews_df = pd.read_csv(reviews_file)
target_levels_df = pd.read_csv(target_levels_file)
target_map = dict(zip(target_levels_df.page_id, target_levels_df.Level))

# Hyperparameters
temperatures = [1.0, 1.2, 1.3]
top_ks = [20, 50]
top_ps = [0.8, 0.9]

# ==================== GENERATE AND SCORE ====================
toxic_rows = []
counter_rows = []
start_time = datetime.now()

for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
    page_id = row['page_id']
    original = row['cleaned_comment']
    level = target_map.get(page_id, 5)

    input_text = original

    for lvl in range(1, level + 1):
        for temp in temperatures:
            for top_k in top_ks:
                for top_p in top_ps:
                    # Toxic Generation
                    tox_prompt = f"Escalate this message: {input_text}"
                    tox_gen = generate_text(tox_prompt, temperature=temp, top_k=top_k, top_p=top_p)
                    toxic_rows.append({
                        'page_id': page_id,
                        'level': lvl,
                        'temperature': temp,
                        'top_k': top_k,
                        'top_p': top_p,
                        'text': tox_gen
                    })
                    torch.cuda.empty_cache()  # ✅ Clear GPU memory after toxic gen

                    # Counter Generation
                    ctr_prompt = f"De-escalate this message: {tox_gen}"
                    ctr_gen = generate_text(ctr_prompt, temperature=temp, top_k=top_k, top_p=top_p)
                    counter_rows.append({
                        'page_id': page_id,
                        'level': lvl,
                        'temperature': temp,
                        'top_k': top_k,
                        'top_p': top_p,
                        'text': ctr_gen
                    })
                    torch.cuda.empty_cache()  # ✅ Clear GPU memory after counter gen

                    input_text = ctr_gen

# Convert to DataFrames and Score
toxic_df = pd.DataFrame(toxic_rows)
counter_df = pd.DataFrame(counter_rows)

# Save only texts
text_only_toxic = toxic_df[['page_id', 'level', 'temperature', 'top_k', 'top_p', 'text']]
text_only_counter = counter_df[['page_id', 'level', 'temperature', 'top_k', 'top_p', 'text']]

text_only_toxic.to_csv(text_only_toxic_file, index=False)
text_only_counter.to_csv(text_only_counter_file, index=False)

# Score separately
scored_toxic = toxic_df.copy()
scored_counter = counter_df.copy()

scored_toxic = scored_toxic.join(scored_toxic.apply(score_text, axis=1))
scored_counter = scored_counter.join(scored_counter.apply(score_text, axis=1))

# Extract only scores
score_cols = ['page_id', 'level', 'temperature', 'top_k', 'top_p', 'hatebert_score', 'neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity', 'stanford_sentiment']
scored_toxic[score_cols].to_csv(scores_only_toxic_file, index=False)
scored_counter[score_cols].to_csv(scores_only_counter_file, index=False)

end_time = datetime.now()
with open(base_path + "generation_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Start Time: {start_time}\n")
    f.write(f"End Time: {end_time}\n")

print("✅ Saved separated text and score outputs for toxic and counter responses.")
