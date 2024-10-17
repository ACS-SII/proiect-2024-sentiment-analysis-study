import tensorflow as tf
import pandas as pd
import numpy as np
import json
import re
import os
import shutil
import datasets


from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from deep_translator import GoogleTranslator
from datasets import Dataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

from sklearn.metrics import accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def prep_doc(doc):
    cleaned = list()
    for i in doc:
        rev = i['title'] + ' . ' + i['content']
        # rev=i['content']
        # rev = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', rev)
        rev = rev.lower()
        rev = re.sub(r'\s+', ' ', rev)
        rev = re.sub(r'(?<! )(?=[.,!?()"\'/:;/\\\[^\]])|(?<=[.,!?()"\'/:;/\\\[^\]])(?! )', r' ', rev)
        cleaned.append(rev)
    return cleaned


def read_data(set):
    pos_data = list()
    neg_data = list()
    for i in set:
        # print(i)
        match i['starRating']:
            case '1':
                neg_data.append(i)
            case '2':
                neg_data.append(i)
            case '4':
                pos_data.append(i)
            case '5':
                pos_data.append(i)
    return neg_data, pos_data


def read_docs():
    f = open('laroseda_train.json')
    # returns JSON object as
    # a dictionary
    train_data = json.load(f)
    f.close()

    f = open('laroseda_test.json')
    # returns JSON object as
    # a dictionary
    test_data = json.load(f)
    # Closing file
    f.close()

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

#
# def preprocess_function(examples):
#     return tokenizer(examples["text"], truncation=True)




def run_test(model_identifier,train_texts,test_texts):

    train_labels = [1] * len(train_pos_clean) + [0] * len(train_neg_clean)
    test_labels = [1] * len(test_pos_clean) + [0] * len(test_neg_clean)

    # Load pre-trained model and tokenizer
    # model_identifier = "dumitrescustefan/bert-base-romanian-cased-v1"
    # model_identifier = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2)  # Adjust num_labels

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=200)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=200)

    # Create Hugging Face Dataset objects using tokenized data
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

    torch.save(model.state_dict(), 'model.pth')

    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy="no",  # Disable saving completely during training
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        eval_strategy="steps",  # Update eval_strategy
        logging_dir="./logs",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Add this line
    )

    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")

    return eval_result


train_pos_clean, train_neg_clean, test_pos_clean, test_neg_clean = read_docs()

train_texts = train_pos_clean + train_neg_clean
train_labels = [1] * len(train_pos_clean) + [0] * len(train_neg_clean)

test_texts = test_pos_clean + test_neg_clean
test_labels = [1] * len(test_pos_clean) + [0] * len(test_neg_clean)

names=['racai/distilbert-base-romanian-uncased',
       'dumitrescustefan/bert-base-romanian-cased-v1',
       'bert-base-multilingual-cased'
       ]

eval_result = run_test(names[0],train_texts, test_texts)



#
# model = AutoModelForSequenceClassification.from_pretrained('nico-dv/sentiment-analysis-model', num_labels=2)  # Adjust num_labels
#
# # nico-dv/sentiment-analysis-model
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics  # Add this line
# )
#
# eval_result = trainer.evaluate()
# print(f"Evaluation results: {eval_result}")