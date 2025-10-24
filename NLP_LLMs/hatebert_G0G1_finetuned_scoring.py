import torch
import logging
import time
import gc
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

logging.basicConfig(level=logging.INFO)

class WeightedHateBERT(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = None  # Not used in inference

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

class HateBERTScorer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        model_path = "C:/FuzzyNov2024/hatebert_finetuned_group1_vs_0_8020"
        config = AutoConfig.from_pretrained(model_path)
        self.model = WeightedHateBERT.from_pretrained(model_path, config=config).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def score_batch(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            return probs[:, 1].tolist()  # Score for class 1 (toxic)

def process_toxicity(input_path, output_path):
    start_time = time.time()

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return

    if not {'rev_id', 'text'}.issubset(df.columns):
        logging.error("Missing required columns.")
        return

    scorer = HateBERTScorer()
    batch_size = 32
    all_scores = []

    logging.info(f"Scoring {len(df)} comments...")

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]
        comments = batch["text"].tolist()
        scores = scorer.score_batch(comments)

        for i, score in enumerate(scores):
            all_scores.append({
                "rev_id": batch.iloc[i]["rev_id"],
                "toxicity_score": score
            })

        logging.info(f"Processed {min(end, len(df))}/{len(df)}")

    result_df = pd.DataFrame(all_scores)
    result_df.to_csv(output_path, index=False)

    logging.info(f"✅ Completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"📝 Results saved to {output_path}")

    gc.collect()
    torch.cuda.empty_cache()

# ==== Run the process ====
if __name__ == "__main__":
    input_file = "C:/FuzzyNov2024/toxicity_project/final_code_data/matched_records_70K.csv"
    output_file = "C:/FuzzyNov2024/toxicity_project/final_code_data/70K_hatebert_finetuned_group1_vs_0_8020_batch_scores.csv"
    process_toxicity(input_file, output_file)
