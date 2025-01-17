import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

DEVICE_MAP = "auto"

# Load and prepare datasets
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

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(llama_df)
eval_dataset = Dataset.from_pandas(llama_df_val)
print(train_dataset)

# PEFT Configuration
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # Example modules for fine-tuning
)



# 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# SFT Configuration (replacing TrainingArguments)
sft_config = SFTConfig(
    output_dir="./results-RoGemma2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    warmup_steps=5,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=3,
    max_grad_norm=0.3,
    warmup_ratio=0,
    group_by_length=True,
    report_to="tensorboard",
    max_seq_length=400,
    dataset_text_field="text",
    packing=True,
)

# Load model and tokenizer
MPATH = "OpenLLM-Ro/RoGemma2-9b-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MPATH,
    quantization_config=bnb_config,
    device_map=DEVICE_MAP,
)
model.config.use_cache = True
model.config.pretraining_tp = 1



for name, param in model.named_parameters():
    print(f"{name}: {param.device}")


tokenizer = AutoTokenizer.from_pretrained(MPATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"




# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,  # Pass SFTConfig instead of TrainingArguments
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained("RoGemma2-9b_fn")
