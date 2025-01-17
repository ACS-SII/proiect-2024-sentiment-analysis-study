
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel
import csv

MPATH = "OpenLLM-Ro/RoLlama2-7b-Instruct"
DEVICE_MAP = "auto"
print("HELLO")

tokenizer = AutoTokenizer.from_pretrained(MPATH)
model_base = AutoModelForCausalLM.from_pretrained(
    MPATH,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=DEVICE_MAP,
)

model2 = PeftModel.from_pretrained(model_base, "RoLlama2-7b_fn").merge_and_unload()


prompt1 = "Experienta cumparaturilor online pe acest site web este una sub asteptarile mele. Comunicarea cu clientul este destul de slaba. Urmarirea comenzii lasa de dorit. Termenul de livrare este destul de nesatisfacator.Calitatea produselor este conform asteptarilor."
prompt2 = "sunt de calitate si sunt si ok ca si pret."

query1 = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|> Esti un asistent util care clasifica recenziile in 'Pozitiv' sau 'Negativ'. Raspunde doar cu 'Pozitiv' sau 'Negativ'.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
""" + prompt1 + """
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
query2 = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|> Esti un asistent util care clasifica recenziile in 'Pozitiv' sau 'Negativ'. Raspunde doar cu 'Pozitiv' sau 'Negativ'.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
""" + prompt2 + """
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
print("start")
pipe = pipeline(task="text-generation", model=model2, tokenizer=tokenizer, max_length = 2000, truncation = True)
result1 = pipe(query1)
res1 = result1[0]['generated_text'].split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|end_of_text|>")[0]
print(res1)
result2 = pipe(query2)
res2 = result2[0]['generated_text'].split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|end_of_text|>")[0]
print(res2)
print("END======")



