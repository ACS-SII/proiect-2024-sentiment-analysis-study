import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import (SFTTrainer, SFTConfig)

DEVICE_MAP = "auto"

train_df = pd.read_csv("../datasets/train_dataset.csv").sample(frac=1, random_state=42).reset_index(drop=True)
eval_df = pd.read_csv("../datasets/val_dataset.csv").sample(frac=1, random_state=42).reset_index(drop=True)
train_df = train_df.dropna(subset=['body', 'label'])
eval_df = eval_df.dropna(subset=['body', 'label'])

llama_df = pd.DataFrame()
llama_df['text'] = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|> Esti un asistent util care clasifica recenziile in 'Pozitiv' sau 'Negativ'. Raspunde doar cu 'Pozitiv' sau 'Negativ'.
<|eot_id|>
<|start_header_id|>user<|end_header_id|> """ + train_df['body'].astype(str) + \
"""<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""" + train_df['label'].astype(str) + "<|end_of_text|>"

llama_df.to_csv("sentiment_dataset.csv", index=False)

llama_df_val = pd.DataFrame()
llama_df_val['text'] = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|> Esti un asistent util care clasifica recenziile in 'Pozitiv' sau 'Negativ'. Raspunde doar cu 'Pozitiv' sau 'Negativ'.
<|eot_id|>
<|start_header_id|>user<|end_header_id|> """ + eval_df['body'].astype(str) + \
"""<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""" + eval_df['label'].astype(str) + "<|end_of_text|>"

llama_df_val.to_csv("sentiment_dataset_eval.csv", index=False)

MPATH = "OpenLLM-Ro/RoMistral-7b-Instruct"
train_dataset = Dataset.from_pandas(llama_df)
eval_dataset = Dataset.from_pandas(llama_df_val)
print(train_dataset)

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # stores weights in 4-bits
    bnb_4bit_quant_type="nf4", # quantization of floats to 4-bit integers
    bnb_4bit_compute_dtype=torch.float16, # compute in half precision
    bnb_4bit_use_double_quant=True # double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    MPATH,
    quantization_config=bnb_config,
#    attn_implementation="flash_attention_2",
    device_map=DEVICE_MAP,
)
model.config.use_cache = True
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MPATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

training_arguments = TrainingArguments(
    output_dir="./results-RoMistral",
    #max_steps=100,
    num_train_epochs=2,
    per_device_train_batch_size=4, # per GPU
    gradient_accumulation_steps=1, # Number of update steps to accumulate the gradients for
    optim="adamw_torch", # Optimizer to use
    save_steps=500,# Save checkpoint every X updates steps
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=3,
    logging_steps=20, # Log every X updates steps
    learning_rate=1.0e-04,
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    fp16=False, # Half precision floats
    bf16=True, # set bf16 to True with an A100
    max_grad_norm=0.3, # gardiet clipping: If a gradient exceeds some threshold value, we clip that gradient to the threshold
    warmup_ratio=0, # Percentage of updates steps to warmup the learning rate over
    group_by_length=True, # Group sequences into batches with same length, saves memory and speeds up training considerably
    report_to="tensorboard" # TensorBoard provides tooling for tracking and visualizing metrics as well as visualizing models
)



trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset= eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    dataset_text_field="text",
    max_seq_length=512,
    packing=True,
)

trainer.train()

trainer.model.save_pretrained("RoMistral-7b_fn")