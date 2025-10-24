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

# ==================== SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# GPT-2 Model (Generation)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model.eval()
torch.cuda.empty_cache()

# GPT_NEO Model (Generation)
gpt_neo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
gpt_neo_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
gpt_neo_tokenizer.pad_token = gpt_neo_tokenizer.eos_token
gpt_neo_model.eval()
torch.cuda.empty_cache()

# MISTRAL Model (Generation)
mistral_model_path = "C:/Users/GPUWIN-1/Mistral-7B-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_path,
    torch_dtype=torch.float16
).to(device)
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
def get_hatebert_score(text):
    try:
        inputs = hatebert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.inference_mode():
            outputs = hatebert_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            return scores[0][1].item()
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

def generate_text(prompt, temperature=1.0, top_k=50, top_p=0.95, max_new_tokens=7):
    inputs = mistral_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    max_len = getattr(mistral_model.config, "n_positions", getattr(mistral_model.config, "max_position_embeddings", 1024))
    if input_len + max_new_tokens > max_len:
        inputs["input_ids"] = inputs["input_ids"][:, -max_len + max_new_tokens:]

    with torch.inference_mode():
        output = mistral_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=mistral_tokenizer.pad_token_id
        )
    return mistral_tokenizer.decode(output[0], skip_special_tokens=True)

def score_text(row):
    scores = {
        "hatebert_score": get_hatebert_score(row['text']),
        **get_vader_score(row['text']),
        **get_textblob_score(row['text'])
    }
    return pd.Series(scores)

# ==================== LOAD DATA ====================
base_path = "C:/FuzzyNov2024/toxicity_project/final_code_data/"
reviews_file = base_path + "group_one_reviews.csv"
target_levels_file = base_path + "target_levels.csv"

text_only_toxic_file = base_path + "text_output_escalated_MISTRAL_G1.csv"
scores_only_toxic_file = base_path + "score_output_escalated_MISTRAL_G1.csv"
text_only_counter_file = base_path + "text_output_counter_MISTRAL_G1.csv"
scores_only_counter_file = base_path + "score_output_counter_MISTRAL_G1.csv"

reviews_df = pd.read_csv(reviews_file)
target_levels_df = pd.read_csv(target_levels_file)
target_map = dict(zip(target_levels_df.page_id, target_levels_df.Level))

# Hyperparameters
temperatures = [1.0, 1.2, 1.3]
top_ks = [20, 50]
top_ps = [0.8, 0.9]

torch.cuda.empty_cache()

# ==================== GENERATE AND SCORE ====================
all_rows = []
counter_rows = []
start_time = datetime.now()

for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
    page_id = row['page_id']
    original = row['cleaned_comment']
    level = target_map.get(page_id, 5)

    for temp in temperatures:
        for top_k in top_ks:
            for top_p in top_ps:
                last_counter_text = original
                for lvl in range(1, level + 1):
                    tox_prompt = f"Escalate this message: {last_counter_text}"
                    tox_gen_full = generate_text(tox_prompt, temperature=temp, top_k=top_k, top_p=top_p)
                    tox_gen = re.sub(r"(?i)Escalate this message:\s*", "", tox_gen_full).strip()

                    ctr_prompt = f"De-escalate this message: {tox_gen}"
                    ctr_gen_full = generate_text(ctr_prompt, temperature=temp, top_k=top_k, top_p=top_p)
                    ctr_gen = re.sub(r"(?i)De-escalate this message:\s*", "", ctr_gen_full).strip()

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

                    last_counter_text = ctr_gen

# Convert to DataFrames
result_df = pd.DataFrame(all_rows)
counter_df = pd.DataFrame(counter_rows)

# Save toxic text output
result_df.to_csv(text_only_toxic_file, index=False)

# Score and save toxic text
result_df['text'] = result_df['toxic'] 
scores_df = result_df[['page_id', 'level', 'temperature', 'top_k', 'top_p', 'text']].copy()
scores_df = scores_df.join(scores_df.apply(score_text, axis=1))
score_cols = ['page_id', 'level',  'hatebert_score', 'neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity']
scores_df[score_cols].to_csv(scores_only_toxic_file, index=False)

# Save counter text
counter_df.to_csv(text_only_counter_file, index=False)

# Score and save counter text
counter_df['text'] = counter_df['counter']
counter_scores_df = counter_df[['page_id', 'level', 'temperature', 'top_k', 'top_p', 'text']].copy()
counter_scores_df = counter_scores_df.join(counter_scores_df.apply(score_text, axis=1))
counter_scores_df[score_cols].to_csv(scores_only_counter_file, index=False)

end_time = datetime.now()
with open(base_path + "generation_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Start Time: {start_time}\n")
    f.write(f"End Time: {end_time}\n")

print("✅ Saved each level in one row with original shown only once. Counter scores and text saved separately.")

