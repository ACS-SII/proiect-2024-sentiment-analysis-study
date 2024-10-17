import openai
import json
import pandas as pd

from sklearn.metrics import accuracy_score

from openai import OpenAI
OPENAI_API_KEY= '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'
client = OpenAI(api_key = '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')

# Set up the OpenAI API key
# client.api_key = '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'


def prep_doc(doc):
    cleaned = list()
    for i in doc:
        rev = i['title'] + ' . ' + i['content']
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


def classify_sentiment_gpt(text):
    client = OpenAI(api_key='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')
    prompt = f"Classify the sentiment of the following Romanian text as Positive or Negative:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can use "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies sentiment."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1
    )
    return response.choices[0].message.content.strip()

# Function to process and label the dataset
def label_and_evaluate(test_pos, test_neg):
    test_texts = test_pos[1] + test_neg[1]
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    # Classify sentiments using GPT
    predictions = [classify_sentiment_gpt(text) for text in test_texts]

    # Map GPT output to numeric labels
    sentiment_mapping = {'Positive': 1, 'Negative': 0}
    predictions_numeric = [sentiment_mapping.get(p, -1) for p in predictions]  # Default to -1 if not found

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions_numeric)

    # Save results to CSV for review
    result_df = pd.DataFrame({'text': test_texts, 'predicted_sentiment': predictions, 'true_label': test_labels})
    result_df.to_csv('labeled_test_results_gpt.csv', index=False)

    return accuracy


def analyze_gpt35(text):
    messages = [
        {"role": "system", "content": """You are trained to analyze and detect the sentiment of given text. 
                                        If you're unsure of an answer, you can say "not sure" and recommend users to review manually."""},
        {"role": "user", "content": f"""Analyze the following product review and determine if the sentiment is: positive or negative. 
                                        Return answer in single word as either positive or negative: {text}"""}
    ]

    client = OpenAI(
        api_key='sk-vRQ03azKdfCEtIX4-WaZen9G0cZT1frbFowNJ3mtu2T3BlbkFJAmd1JsBhYGKg-h06TTltXtetNomlazBOm6ezqMwNQA')

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0)

    response_text = response.choices[0].message.content.strip().lower()

    return response_text


train_pos_clean, train_neg_clean, test_pos_clean, test_neg_clean=read_docs()


rezz=analyze_gpt35(train_pos_clean[5])


# Example usage
accuracy = label_and_evaluate(test_pos_clean, test_neg_clean)
print(f"Accuracy: {accuracy:.2f}")

