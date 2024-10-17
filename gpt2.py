import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score
import json
import re
import os
import numpy as np

# Ensure TensorFlow doesn't interfere with PyTorch
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def prep_doc(doc):
    cleaned = []
    for i in doc:
        rev = i['title'] + ' . ' + i['content']
        rev = rev.lower()
        rev = re.sub(r'\s+', ' ', rev)
        rev = re.sub(r'(?<! )(?=[.,!?()"\'/:;/\\\[^\]])|(?<=[.,!?()"\'/:;/\\\[^\]])(?! )', r' ', rev)
        cleaned.append(rev)
    return cleaned

def read_data(set):
    pos_data = []
    neg_data = []
    for i in set:
        if i['starRating'] in ['1', '2']:
            neg_data.append(i)
        elif i['starRating'] in ['4', '5']:
            pos_data.append(i)
    return neg_data, pos_data

def read_docs():
    with open('laroseda_train.json') as f:
        train_data = json.load(f)
    with open('laroseda_test.json') as f:
        test_data = json.load(f)

    train_neg, train_pos = read_data(train_data['reviews'])
    test_neg, test_pos = read_data(test_data['reviews'])

    train_pos_clean = prep_doc(train_pos)
    train_neg_clean = prep_doc(train_neg)
    test_pos_clean = prep_doc(test_pos)
    test_neg_clean = prep_doc(test_neg)

    return train_pos_clean, train_neg_clean, test_pos_clean, test_neg_clean

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


def run_test(model_identifier, train_texts, test_texts, train_labels, test_labels):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    # Add a new padding token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2)

    # Resize token embeddings after adding new special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Set the padding token ID in the model configuration
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize the texts with padding and truncation
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=200)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=200)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'label': train_labels
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'label': test_labels
    })

    # Set the format of the datasets for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy="no",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")

    return eval_result


# Load and preprocess data
train_pos_clean, train_neg_clean, test_pos_clean, test_neg_clean = read_docs()

train_texts = train_pos_clean + train_neg_clean
train_labels = [1] * len(train_pos_clean) + [0] * len(train_neg_clean)

test_texts = test_pos_clean + test_neg_clean
test_labels = [1] * len(test_pos_clean) + [0] * len(test_neg_clean)

names = [
    # 'gpt2',  # The standard GPT-2 model identifier
'DGurgurov/xlm-r_romanian_sentiment'
]

# Run the test with GPT-2
eval_result = run_test(names[0], train_texts, test_texts, train_labels, test_labels)
