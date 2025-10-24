import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
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

def generate_text(prompt, max_new_tokens=25):
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
            do_sample=True,
            temperature=1.7,
            top_k=40,
            top_p=0.85,
            repetition_penalty=1.8,
            pad_token_id=mistral_tokenizer.pad_token_id,
            eos_token_id=mistral_tokenizer.eos_token_id
        )
    decoded = mistral_tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.replace(prompt, '').strip()

prompt_cleaner = re.compile(r"--\\s*(escalate|respond).*?(--|$)", re.IGNORECASE)
def clean_generation(text):
    return prompt_cleaner.sub("", text).strip()

def extract_keywords(text, top_n=5):
    blob = TextBlob(text)
    keywords = [word.lower() for word, tag in blob.tags if tag.startswith('NN') or tag.startswith('VB')]
    return list(set(keywords))[:top_n]

def keyword_overlap(keywords, response, min_match=1):
    response_words = set(response.lower().split())
    matches = sum(1 for k in keywords if k in response_words)
    return matches >= min_match

def enrich_prompt(prompt, keywords):
    return prompt + "\nEnsure the response includes these concepts: " + ", ".join(keywords[:3])

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

# ==================== GENERATE AND SCORE ====================
all_rows = []
counter_rows = []
start_time = datetime.now()

for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
    page_id = row['page_id']
    original = row['cleaned_comment']
    level = target_map.get(page_id, 3)

    orig_keywords = extract_keywords(original)
    toxic_variants = [original]
    counter_variants = []

    for lvl in range(1, level + 1):
        n_context = max(1, min(lvl + 2, 4))
        combined_pool = toxic_variants + counter_variants + [original]

        toxic_generated = False
        for _ in range(10):
            toxic_context = "\n".join(random.sample(combined_pool, min(n_context, len(combined_pool))))
            toxic_prompt = enrich_prompt(
                f"Continuing the discussion (keywords: {', '.join(orig_keywords)}):\n{toxic_context}\n-- escalate this with a more toxic tone, reacting aggressively to the ideas above.",
                orig_keywords)
            gen_full = generate_text(toxic_prompt, max_new_tokens=random.randint(20, 35))
            gen = clean_generation(gen_full)
            if gen.strip() and gen not in toxic_variants and keyword_overlap(orig_keywords, gen):
                toxic_variants.append(gen)
                all_rows.append({
                    'page_id': page_id,
                    'level': lvl,
                    'original': original if lvl == 1 else '',
                    'toxic': gen
                })
                toxic_generated = True
                break

        if not toxic_generated:
            fallback_gen = generate_text(f"React to this comment in a harsh, critical, or rude way: {original}", max_new_tokens=25)
            fallback_clean = clean_generation(fallback_gen)
            toxic_variants.append(fallback_clean)
            all_rows.append({
                'page_id': page_id,
                'level': lvl,
                'original': original if lvl == 1 else '',
                'toxic': fallback_clean
            })

        counter_generated = False
        if len(toxic_variants) > 1:
            latest = toxic_variants[-1]
            for _ in range(10):
                counter_context = "\n".join([latest] + random.sample(toxic_variants[:-1], min(n_context - 1, len(toxic_variants) - 1)))
                counter_prompt = enrich_prompt(
                    f"Continuing the discussion (keywords: {', '.join(orig_keywords)}):\n{counter_context}\n-- respond calmly to de-escalate the aggression above constructively.",
                    orig_keywords)
                gen_full = generate_text(counter_prompt, max_new_tokens=random.randint(20, 35))
                gen = clean_generation(gen_full)
                if gen.strip() and gen not in counter_variants and keyword_overlap(orig_keywords, gen):
                    counter_variants.append(gen)
                    counter_rows.append({
                        'page_id': page_id,
                        'level': lvl,
                        'counter': gen
                    })
                    counter_generated = True
                    break

        if not counter_generated:
            fallback_counter = generate_text(f"Respond in a calm, respectful, and de-escalating way to: {latest}", max_new_tokens=25)
            fallback_clean = clean_generation(fallback_counter)
            counter_variants.append(fallback_clean)
            counter_rows.append({
                'page_id': page_id,
                'level': lvl,
                'counter': fallback_clean
            })

        torch.cuda.empty_cache()

result_df = pd.DataFrame(all_rows)
counter_df = pd.DataFrame(counter_rows)

result_df.to_csv(text_only_toxic_file, index=False)
result_df['text'] = result_df['toxic']
scores_df = pd.concat([result_df[['page_id', 'level']], batch_score_texts(result_df['text'].tolist())], axis=1)
scores_df.to_csv(scores_only_toxic_file, index=False)

counter_df.to_csv(text_only_counter_file, index=False)
counter_df['text'] = counter_df['counter']
counter_scores_df = pd.concat([counter_df[['page_id', 'level']], batch_score_texts(counter_df['text'].tolist())], axis=1)
counter_scores_df.to_csv(scores_only_counter_file, index=False)

end_time = datetime.now()
with open(base_path + "generation_summary.txt", "a", encoding="utf-8") as f:
    f.write(f"Final Generation End Time: {end_time}\n")

print("✅ Saved toxic and counter responses with batch scoring.")
