import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ====== Setup ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GPT-2 model and tokenizer
gpt_model_name = "gpt2"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name).to(device)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model.config.pad_token_id = gpt_tokenizer.eos_token_id
gpt_model.eval()

# Load HateBERT model for toxicity scoring
hatebert_model_path = "C:/FuzzyNov2024/hatebert_finetuned"
hatebert_model = AutoModelForSequenceClassification.from_pretrained(hatebert_model_path).to(device)
hatebert_tokenizer = AutoTokenizer.from_pretrained(hatebert_model_path)
hatebert_model.eval()

# Load input data
reviews_file = 'C:/FuzzyNov2024/toxicity_project/final_code_data/group_one_reviews.csv'
target_levels_file = 'C:/FuzzyNov2024/toxicity_project/final_code_data/target_levels.csv'
output_toxic = 'C:/FuzzyNov2024/toxicity_project/final_code_data/output_escalated.csv'
output_counter = 'C:/FuzzyNov2024/toxicity_project/final_code_data/output_deescalated.csv'

reviews_df = pd.read_csv(reviews_file)
target_levels_df = pd.read_csv(target_levels_file)
target_map = dict(zip(target_levels_df.page_id, target_levels_df.Level))

# Hyperparameters
temperatures = [1.0, 1.2, 1.3]
top_ks = [20, 50]
top_ps = [0.8, 0.9]

# Functions
def get_toxicity_score(text):
    try:
        inputs = hatebert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = hatebert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            return probs[0][1].item()  # Class 1 = toxic
    except:
        return 0.0

def generate_response(prompt, temperature=1.0, top_k=50, top_p=0.9, max_new_tokens=10):
    inputs = gpt_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    if inputs.input_ids.shape[1] > gpt_model.config.n_positions - max_new_tokens:
        inputs.input_ids = inputs.input_ids[:, -gpt_model.config.n_positions + max_new_tokens:]

    output = gpt_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=max(top_k, 1),
        top_p=max(top_p, 0.1),
        temperature=temperature,
        pad_token_id=gpt_tokenizer.pad_token_id
    )
    return gpt_tokenizer.decode(output[0], skip_special_tokens=True)

# Run Generation
escalation_rows = []
deescalation_rows = []

for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
    page_id = row['page_id']
    original_text = row['cleaned_comment']
    level = target_map.get(page_id, 5)  # Default to 5 if not found

    escalation_rows.append({
        'page_id': page_id,
        'level': 0,
        'temperature': '',
        'top_k': '',
        'top_p': '',
        'toxicity_score': get_toxicity_score(original_text),
        'text': original_text
    })

    input_text = original_text

    for lvl in range(1, level + 1):
        for temp in temperatures:
            for top_k in top_ks:
                for top_p in top_ps:
                    toxic_prompt = f"Escalate this message: {input_text}"
                    toxic_gen = generate_response(toxic_prompt, temperature=temp, top_k=top_k, top_p=top_p)
                    toxic_score = get_toxicity_score(toxic_gen)

                    escalation_rows.append({
                        'page_id': page_id,
                        'level': lvl,
                        'temperature': temp,
                        'top_k': top_k,
                        'top_p': top_p,
                        'toxicity_score': toxic_score,
                        'text': toxic_gen
                    })

                    # De-escalate
                    counter_prompt = f"De-escalate this message: {toxic_gen}"
                    counter_gen = generate_response(counter_prompt, temperature=temp, top_k=top_k, top_p=top_p)
                    counter_score = get_toxicity_score(counter_gen)

                    deescalation_rows.append({
                        'page_id': page_id,
                        'level': lvl,
                        'temperature': temp,
                        'top_k': top_k,
                        'top_p': top_p,
                        'toxicity_score': counter_score,
                        'text': counter_gen
                    })

                    input_text = counter_gen  # Continue from de-escalated response

# Save to CSV
pd.DataFrame(escalation_rows).to_csv(output_toxic, index=False)
pd.DataFrame(deescalation_rows).to_csv(output_counter, index=False)

print("✅ Generation and scoring complete.")
