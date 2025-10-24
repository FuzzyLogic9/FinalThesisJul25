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

        import stanza
        stanza.download('en')
        self.stanza_pipeline = stanza.Pipeline('en', processors='tokenize,sentiment', use_gpu=torch.cuda.is_available())

        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score_batch(self, texts):
        from pattern3.en import sentiment as pattern_sentiment
        from textblob import TextBlob

        results = []
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
            toxicity_scores = probs[:, 1].tolist()

        pattern_results = []
        for t in texts:
            try:
                pattern_results.append(pattern_sentiment(t))
            except:
                pattern_results.append((0.0, 0.0))

        tb_results = [(TextBlob(t).sentiment.polarity, TextBlob(t).sentiment.subjectivity) for t in texts]
        vader_results = [self.vader.polarity_scores(t) for t in texts]
        stanza_results = []
        for t in texts:
            try:
                r = self.stanza_pipeline(t)
                s = r.sentences[0].sentiment if r.sentences else 1
            except:
                s = 1
            stanza_results.append(s)

        for idx, (text, score) in enumerate(zip(texts, toxicity_scores)):
            pattern_polarity, pattern_subjectivity = pattern_results[idx]
            tb_polarity, tb_subjectivity = tb_results[idx]
            vader_result = vader_results[idx]
            stanza_sentiment = stanza_results[idx]

            results.append({
                'toxicity_score': score,
                'pattern_polarity': pattern_polarity,
                'pattern_subjectivity': pattern_subjectivity,
                'tb_polarity': tb_polarity,
                'tb_subjectivity': tb_subjectivity,
                'vader_neg': vader_result['neg'],
                'vader_neu': vader_result['neu'],
                'vader_pos': vader_result['pos'],
                'vader_compound': vader_result['compound'],
                'stanza_sentiment': stanza_sentiment
            })

        return results

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

        for i, score_data in enumerate(scores):
            all_scores.append({
                "rev_id": batch.iloc[i]["rev_id"],
                "toxicity_score": score_data["toxicity_score"],
                "pattern_polarity": score_data["pattern_polarity"],
                "pattern_subjectivity": score_data["pattern_subjectivity"],
                "tb_polarity": score_data["tb_polarity"],
                "tb_subjectivity": score_data["tb_subjectivity"],
                "vader_neg": score_data["vader_neg"],
                "vader_neu": score_data["vader_neu"],
                "vader_pos": score_data["vader_pos"],
                "vader_compound": score_data["vader_compound"],
                "stanza_sentiment": score_data["stanza_sentiment"]
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
