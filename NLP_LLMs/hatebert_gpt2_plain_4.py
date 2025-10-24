import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from datetime import datetime
import re
import random

# ==================== SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# MISTRAL Model (Generation) with half precision and auto device mapping
mistral_model_path = "C:/Users/GPUWIN-1/Mistral-7B-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
mistral_model.eval()
torch.cuda.empty_cache()

# HateBERT (Toxicity Detection)
hatebert_path = "C:/FuzzyNov2024/hatebert_finetuned"
hatebert_tokenizer = AutoTokenizer.from_pretrained(hatebert_path)
hatebert_model = AutoModelForSequenceClassification.from_pretrained(hatebert_path).to(device)
hatebert_model.eval()
torch.cuda.empty_cache()

# VADER
vader = SentimentIntensityAnalyzer()

# ==================== FUNCTIONS ====================
def batch_score_texts(texts):
    hate_scores = []
    vader_scores = []
    tb_scores = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        try:
            inputs = hatebert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = hatebert_model(**inputs)
                batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].tolist()
        except:
            batch_scores = [0.0] * len(batch)
        hate_scores.extend(batch_scores)

        for t in batch:
            vader_scores.append(vader.polarity_scores(t))
            tb = TextBlob(t)
            tb_scores.append((tb.polarity, tb.subjectivity))

    return pd.DataFrame({
        'hatebert_score': hate_scores,
        'neg': [s['neg'] for s in vader_scores],
        'neu': [s['neu'] for s in vader_scores],
        'pos': [s['pos'] for s in vader_scores],
        'compound': [s['compound'] for s in vader_scores],
        'polarity': [s[0] for s in tb_scores],
        'subjectivity': [s[1] for s in tb_scores]
    })

def generate_text(prompt, max_new_tokens=5):
    inputs = mistral_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    max_len = getattr(mistral_model.config, "n_positions", getattr(mistral_model.config, "max_position_embeddings", 1024))

    if input_len + max_new_tokens > max_len:
        inputs["input_ids"] = inputs["input_ids"][:, -max_len + max_new_tokens:]

    with torch.no_grad():
        output = mistral_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            pad_token_id=mistral_tokenizer.pad_token_id
        )
    return mistral_tokenizer.decode(output[0], skip_special_tokens=True)

# ==================== LOAD DATA ====================
base_path = "C:/FuzzyNov2024/toxicity_project/final_code_data/"
reviews_file = base_path + "group_one_reviews_snap.csv"
target_levels_file = base_path + "target_levels_snap.csv"

text_only_toxic_file = base_path + "text_output_escalated_MISTRAL_G1_snap.csv"
scores_only_toxic_file = base_path + "score_output_escalated_MISTRAL_G1_snap.csv"
text_only_counter_file = base_path + "text_output_counter_MISTRAL_G1_snap.csv"
scores_only_counter_file = base_path + "score_output_counter_MISTRAL_G1_snap.csv"

reviews_df = pd.read_csv(reviews_file)
target_levels_df = pd.read_csv(target_levels_file)
target_map = dict(zip(target_levels_df.page_id, target_levels_df.Level))

# Hyperparameters
hyperparams = [(1.0, 20, 0.8)]  # trimmed to a single setting for faster test loop

# ==================== GENERATE AND SCORE ====================
all_rows = []
counter_rows = []
start_time = datetime.now()

for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
    page_id = row['page_id']
    original = row['cleaned_comment']
    level = target_map.get(page_id, 3)

    for temp, top_k, top_p in hyperparams:
        responses = [original]
        for lvl in range(1, level + 1):
            random_idx = random.randint(0, len(responses) - 1)
            reference_comment = responses[random_idx]

            tox_prompt = f"Reply to the original comment '{original}' and also to this with a stronger tone: {reference_comment}"
            tox_gen_full = generate_text(tox_prompt)
            tox_gen = re.sub(r"(?i)Reply to .* with a stronger tone:\s*", "", tox_gen_full).strip()

            ctr_prompt = f"Reply to the original comment '{original}' and also to this with a calmer tone: {tox_gen}"
            ctr_gen_full = generate_text(ctr_prompt)
            ctr_gen = re.sub(r"(?i)Reply to .* with a calmer tone:\s*", "", ctr_gen_full).strip()

            all_rows.append({
                'page_id': page_id,
                'level': lvl,
                'temperature': temp,
                'top_k': top_k,
                'top_p': top_p,
                'original': original if lvl == 1 else '',
                'toxic': tox_gen
            })

            counter_rows.append({
                'page_id': page_id,
                'level': lvl,
                'temperature': temp,
                'top_k': top_k,
                'top_p': top_p,
                'counter': ctr_gen
            })

            responses.append(ctr_gen)
            torch.cuda.empty_cache()

# Convert to DataFrames
result_df = pd.DataFrame(all_rows)
counter_df = pd.DataFrame(counter_rows)

# Save toxic text output
result_df.to_csv(text_only_toxic_file, index=False)

# Score and save toxic text
result_df['text'] = result_df['toxic'] 
scores_df = result_df[['page_id', 'level', 'temperature', 'top_k', 'top_p']].copy()
scores_df = pd.concat([scores_df, batch_score_texts(result_df['text'].tolist())], axis=1)
score_cols = ['page_id', 'level',  'hatebert_score', 'neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity']
scores_df[score_cols].to_csv(scores_only_toxic_file, index=False)

# Save counter text
counter_df.to_csv(text_only_counter_file, index=False)

# Score and save counter text
counter_df['text'] = counter_df['counter']
counter_scores_df = counter_df[['page_id', 'level', 'temperature', 'top_k', 'top_p']].copy()
counter_scores_df = pd.concat([counter_scores_df, batch_score_texts(counter_df['text'].tolist())], axis=1)
counter_scores_df[score_cols].to_csv(scores_only_counter_file, index=False)

end_time = datetime.now()
with open(base_path + "generation_summary.txt", "a", encoding="utf-8") as f:
    f.write(f"Final Generation End Time: {end_time}\n")

print("✅ Saved toxic and counter responses with batch scoring.")
