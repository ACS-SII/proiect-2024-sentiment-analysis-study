from transformers import pipeline
from sklearn.metrics import classification_report
import pandas as pd

# Încarcă modelul și tokenizer-ul fine-tunate
model_path = "llama3-8b_fn"
tokenizer_path = "llama3-8b_fn"
sentiment_pipeline = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer_path,
    device=0  # GPU
)

# Citește setul de test
test_df = pd.read_csv("test_dataset.csv")

# Funcție pentru predicții
def predict_sentiment(review):
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> {review} <|eot_id|>"
    output = sentiment_pipeline(prompt, max_new_tokens=10, num_return_sequences=1)
    prediction = output[0]["generated_text"]
    if "Pozitiv" in prediction:
        return "Pozitiv"
    elif "Negativ" in prediction:
        return "Negativ"
    else:
        return "Unknown"

# Facem predicții pentru fiecare recenzie din setul de test
test_df["predicted_label"] = test_df["body"].apply(predict_sentiment)

# Evaluăm performanța modelului
print(classification_report(test_df["label"], test_df["predicted_label"], target_names=["Pozitiv", "Negativ"]))
