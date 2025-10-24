from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("C:/FuzzyNov2024/toxicity_project/jigsaw")

# Load HateBERT model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

# Remove rows where 'comment_text' is None
dataset = dataset.filter(lambda example: example['comment_text'] is not None)

# Preprocess function
def preprocess_function(examples):
    texts = [str(text) for text in examples["comment_text"]]
    return tokenizer(texts, truncation=True, padding=True)

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Rename 'toxic' column to 'labels'
tokenized_dataset = tokenized_dataset.rename_column("toxic", "labels")

# Remove unused columns
tokenized_dataset = tokenized_dataset.remove_columns(["id", "comment_text", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

# Split dataset into training and validation
if 'train' in tokenized_dataset:
    dataset_split = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    print("Training and validation sets created.")
else:
    print("The 'train' split is not available in the dataset.")

# Training arguments
training_args = TrainingArguments(
    output_dir="hatebert_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=500,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    remove_unused_columns=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune HateBERT
trainer.train()

# Save fine-tuned model
model.save_pretrained("C:/FuzzyNov2024/hatebert_finetuned")
tokenizer.save_pretrained("C:/FuzzyNov2024/hatebert_finetuned")
