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
import stanza
from pattern3.en import sentiment as pattern_sentiment
from tqdm import tqdm
from datetime import datetime
import re
import random

# ==================== SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

stanza.download('en')
stanza_pipeline = stanza.Pipeline('en', processors='tokenize,sentiment', use_gpu=torch.cuda.is_available())

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

hatebert_path = "C:/FuzzyNov2024/hatebert_finetuned"
hatebert_tokenizer = AutoTokenizer.from_pretrained(hatebert_path)
hatebert_model = AutoModelForSequenceClassification.from_pretrained(hatebert_path).to(device)
hatebert_model.eval()
torch.cuda.empty_cache()

vader = SentimentIntensityAnalyzer()

# ==================== FUNCTIONS ====================
def extract_keywords(text, top_n=5):
    try:
        blob = TextBlob(text)
        keywords = [word.lower() for word, tag in blob.tags if tag.startswith('NN') or tag.startswith('VB')]
        return list(set(keywords))[:top_n]
    except:
        return []

def keyword_overlap(keywords, response, min_match=2):
    response_words = set(re.findall(r'\w+', response.lower()))
    return sum(1 for k in keywords if k in response_words) >= min_match

def batch_score_texts(texts):
    hate_scores, vader_scores, tb_scores, stanza_scores, pattern_scores = [], [], [], [], []
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
            try:
                doc = stanza_pipeline(t)
                stanza_scores.append(doc.sentences[0].sentiment if doc.sentences else 1)
            except:
                stanza_scores.append(1)
            try:
                pattern_scores.append(pattern_sentiment(t))
            except:
                pattern_scores.append((0.0, 0.0))

    return pd.DataFrame({
        'hatebert_score': hate_scores,
        'neg': [s['neg'] for s in vader_scores],
        'neu': [s['neu'] for s in vader_scores],
        'pos': [s['pos'] for s in vader_scores],
        'compound': [s['compound'] for s in vader_scores],
        'polarity': [s[0] for s in tb_scores],
        'subjectivity': [s[1] for s in tb_scores],
        'stanza_sentiment': stanza_scores,
        'pattern_polarity': [s[0] for s in pattern_scores],
        'pattern_subjectivity': [s[1] for s in pattern_scores]
    })

def generate_text(prompt, max_new_tokens=None):
    if max_new_tokens is None:
        max_new_tokens = random.randint(15, 30)
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
    return mistral_tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, '').strip()

def clean_generation(text):
    text = re.sub(r"--\\s*(escalate|respond).*?(--|$)", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^\w\s.,!?'-]", "", text).strip()

def improved_enrich_prompt(prompt, keywords):
    return f"{prompt}\nYou must include and respond to at least two of the following terms: {', '.join(keywords[:5])}\nStyle Guidance: Use sarcasm, insults, or accusatory tone without physical threats."

# ==================== TOXIC-ONLY GENERATION ====================
base_path = "C:/FuzzyNov2024/toxicity_project/final_code_data/"
reviews_df = pd.read_csv(base_path + "group_one_reviews.csv")
target_levels_df = pd.read_csv(base_path + "target_levels.csv")
target_map = dict(zip(target_levels_df.page_id, target_levels_df.Level))

text_only_toxic_file = base_path + "text_output_escalated_MISTRAL_G1_OT.csv"
scores_only_toxic_file = base_path + "score_output_escalated_MISTRAL_G1_OT.csv"

all_rows = []
start_time = datetime.now()

id_pattern = re.compile(r"\b\d{5,}\b|votes?[-:]|comments?", re.IGNORECASE)

for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
    page_id = row['page_id']
    original = row['cleaned_comment']
    level = target_map.get(page_id, 3)

    orig_keywords = extract_keywords(original)
    toxic_variants = [original]

    for lvl in range(1, level + 1):
        n_context = max(1, min(lvl + 2, 5))
        combined_pool = [original] + toxic_variants[-2:]

        toxic_generated = False
        for _ in range(10):
            context = "\n".join(combined_pool[:n_context])
            toxic_prompt = improved_enrich_prompt(
                f"Continuing the discussion (keywords: {', '.join(orig_keywords)}):\n{context}",
                orig_keywords)
            gen = clean_generation(generate_text(toxic_prompt, random.randint(20, 35)))
            if (len(gen.split()) >= 8 and gen not in toxic_variants
                    and keyword_overlap(orig_keywords, gen)
                    and not id_pattern.search(gen)):
                toxic_variants.append(gen)
                all_rows.append({
                    'page_id': page_id, 'level': lvl, 'original': original if lvl == 1 else '',
                    'toxic': gen, 'fallback': False
                })
                toxic_generated = True
                break

        if not toxic_generated:
            fallback = clean_generation(generate_text(f"React in a harsh, sarcastic, or rude way to this comment: {original}", 25))
            toxic_variants.append(fallback)
            all_rows.append({
                'page_id': page_id, 'level': lvl, 'original': original if lvl == 1 else '',
                'toxic': fallback, 'fallback': True
            })

        torch.cuda.empty_cache()

result_df = pd.DataFrame(all_rows)
result_df.to_csv(text_only_toxic_file, index=False)
result_df['text'] = result_df['toxic']
scores_df = pd.concat([result_df[['page_id', 'level']], batch_score_texts(result_df['text'].tolist())], axis=1)
scores_df.to_csv(scores_only_toxic_file, index=False)

end_time = datetime.now()
with open(base_path + "generation_summary_ot.txt", "a", encoding="utf-8") as f:
    f.write(f"Toxicity-Only Generation Start Time: {start_time}\n")
    f.write(f"Toxicity-Only Generation End Time: {end_time}\n")
    f.write(f"Duration: {end_time - start_time}\n")

print("\u2705 Saved toxicity-only responses with batch scoring.")
