import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the dataset
dataset_path = "C:/FuzzyNov2024/toxicity_project/jigsaw/train/train.csv"
df = pd.read_csv(dataset_path)

# Define a function to update labels
def update_labels(row):
    # List of columns indicating negative behavior
    negative_labels = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # If any of these labels are 1, set 'toxic' to 1
    if any(row[label] == 1 for label in negative_labels):
        return 1
    else:
        return 0

# Apply the function to update the 'toxic' column
df['labels'] = df.apply(update_labels, axis=1)

# Keep only necessary columns
df = df[['comment_text', 'labels']]

# Remove rows where 'comment_text' is NaN or empty
df = df.dropna(subset=['comment_text'])
df = df[df['comment_text'].str.strip().astype(bool)]


# Reset index to avoid index-related columns
df = df.reset_index(drop=True)

# Remove any unnecessary columns
if '__index_level_0__' in df.columns:
    df = df.drop(columns=['__index_level_0__'])

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)


# Load HateBERT model and tokenizer
model_name = "GroNLP/hateBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda")

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["comment_text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Define the training arguments
training_args = TrainingArguments(
    output_dir="hatebert_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=500,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

# Define a function to compute evaluation metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("C:/FuzzyNov2024/hatebert_finetuned")
tokenizer.save_pretrained("C:/FuzzyNov2024/hatebert_finetuned")

# Update model configuration for label mapping
model.config.id2label = {0: 'non-toxic', 1: 'toxic'}
model.config.label2id = {'non-toxic': 0, 'toxic': 1}

# Evaluation function for test sentences
def evaluate_sentences(sentences, model, tokenizer, threshold=0.5):
    model.eval()
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)
        for sentence, prediction, confidence in zip(sentences, predictions, confidences):
            label = model.config.id2label[prediction.item()]
            print(f"Sentence: '{sentence}'\nPredicted label: {label} (Confidence: {confidence:.4f})\n")

# Test the model with sample sentences
test_sentences = [
    "I hate you, you are terrible!",
    "You are stupid and worthless.",
    "Have a nice day, hope you're well!",
    "I love this!",
    "This is the worst thing ever."
]

evaluate_sentences(test_sentences, model, tokenizer)
