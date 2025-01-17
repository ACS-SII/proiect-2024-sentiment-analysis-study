from sklearn.metrics import classification_report
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel
import csv
import time
import matplotlib.pyplot as plt
import os

MPATH = "OpenLLM-Ro/RoMistral-7b-Instruct"
DEVICE_MAP = "auto"
print("HELLO")

# Inițializare tokenizer și model
tokenizer = AutoTokenizer.from_pretrained(MPATH)
model_base = AutoModelForCausalLM.from_pretrained(
    MPATH,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=DEVICE_MAP,
)

model = PeftModel.from_pretrained(model_base, "RoMistral-7b_fn").merge_and_unload()

sentiment_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000, truncation=True)

# Citire dataset de test
test_df = pd.read_csv("../datasets/test_dataset.csv")

# Funcție pentru a genera predicții
def predict_sentiment(review):
    review = str(review) if review is not None else ""
    
    prompt = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|> Esti un asistent util care clasifica recenziile in 'Pozitiv' sau 'Negativ'. Raspunde doar cu 'Pozitiv' sau 'Negativ'.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
""" + review + """
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    output = sentiment_pipeline(prompt)
    prediction = output[0]["generated_text"].split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|end_of_text|>")[0]
    if "Pozitiv" in prediction:
        return "Pozitiv"
    elif "Negativ" in prediction:
        return "Negativ"
    else:
        return "Unknown"

# Începe măsurarea timpului
start_time = time.time()

# Generăm predicții pentru fiecare recenzie din setul de test
test_df["predicted_label"] = test_df["body"].apply(predict_sentiment)

# Evaluăm performanța modelului
classification_rep = classification_report(test_df["label"], test_df["predicted_label"], target_names=["Pozitiv", "Negativ"], output_dict=True)
print(classification_report(test_df["label"], test_df["predicted_label"], target_names=["Pozitiv", "Negativ"]))

# Calculăm timpul de execuție
end_time = time.time()
execution_time = end_time - start_time

output_dir = "evaluation"
os.makedirs(output_dir, exist_ok=True)

# Salvăm raportul de clasificare și timpul într-un fișier CSV
evaluation_file = os.path.join(output_dir, "evaluation_results.csv")
with open(evaluation_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Pozitiv", "Negativ", "Accuracy"])
    for metric, values in classification_rep.items():
        if isinstance(values, dict):
            writer.writerow([metric, values.get("precision", ""), values.get("recall", ""), values.get("f1-score", "")])
        else:
            writer.writerow([metric, "", "", values])
    writer.writerow([])
    writer.writerow(["Execution Time (s)", execution_time])

print(f"Evaluation results saved to {evaluation_file}")



# Plot precizie, recall și f1-score pentru fiecare clasă
metrics = ["precision", "recall", "f1-score"]
classes = ["Pozitiv", "Negativ"]

for metric in metrics:
    values = [classification_rep[cls][metric] for cls in classes]
    plt.figure(figsize=(8, 6))
    plt.bar(classes, values, color=["#1f77b4", "#ff7f0e"])
    plt.title(f"{metric.capitalize()} per Class")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Class")
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=12)
    plt.savefig(os.path.join(output_dir, f"{metric}_plot.png"))
    plt.close()

# Plot acuratețe globală
accuracy = classification_rep["accuracy"]
plt.figure(figsize=(8, 6))
plt.bar(["Accuracy"], [accuracy], color=["#2ca02c"])
plt.title("Overall Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.text(0, accuracy + 0.02, f"{accuracy:.2f}", ha="center", fontsize=12)
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.close()

print(f"Plots saved to {output_dir}")
