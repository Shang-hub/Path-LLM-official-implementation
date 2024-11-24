
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from peft import LoraConfig, PeftConfig
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import datasets
import torch
import random
import os
import wandb
from datetime import datetime
import math
import pickle as pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import warnings
import torch.nn.functional as F
import torch as th
import pickle as pickle

warnings.filterwarnings("ignore")

# device_ids = [0, 2, 4, 7]
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

wandb.login()
wand_project = "Path-LLM"
if len(wand_project)>0:
    os.environ["WANDB_PROJECT"] = wand_project


dataset = load_dataset("text", data_files={"train": './training_data.txt'})



access_token = access_token

model = "meta-llama/Llama-2-7b-hf"
# Load model directly

tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = dataset.map(tokenize_function, remove_columns=["text"])


print(tokenized_datasets)



model = AutoModelForCausalLM.from_pretrained(model, 
                                             device_map="auto",
                                             token=access_token)

run_name = "graph_representation" + wand_project


# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM"
)


train_params = TrainingArguments(
    output_dir="model/training_process_checkpoint",
    save_strategy="epoch", 
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb",
    run_name = f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=1024,
    args=train_params,
    packing = False,
)

trainer.train()

# Save Model
trainer.save_model("Path-LLM")

