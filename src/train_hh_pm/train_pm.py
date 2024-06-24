from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, TrainingArguments
from accelerate import Accelerator
from trl import RewardConfig, RewardTrainer
from peft import LoraConfig, TaskType
import os
from huggingface_hub import login
from accelerate import init_empty_weights
import wandb
import warnings
import torch.distributed as dist
def is_main_process():
    if False:
        return True
    return dist.get_rank() == 0
if __name__ == "__main__":
    parser = HfArgumentParser(RewardConfig)
    # Add custom arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B") #"meta-llama/Llama-2-7b-hf"
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=32)
    parser.add_argument("--principle", type=str, default=None)
    parser.add_argument("--LoRA", type=str, default="False")
    parser.add_argument("--LoRA_r", type=int, default=None)
    parser.add_argument("--LoRA_alpha", type=int, default=None)
    parser.add_argument("--LoRA_dropout", type=float, default=None)
    parser.add_argument("--margin", type=str, default="True")
    # Parse the dictionary into RewardConfig
    reward_config, config = parser.parse_args_into_dataclasses()
    


    #print(reward_config)
    #reward_config.gradient_checkpointing_kwargs={"use_reentrant":False}

    train_dataset = load_dataset('json', data_files=f'data/hh_labels/hh_test_1_1.jsonl')
    test_dataset = load_dataset('json', data_files=f'data/hh_labels/hh_test_1_1.jsonl') 

    if config.LoRA == "True":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=config.LoRA_r,
            lora_alpha=config.LoRA_alpha,
            lora_dropout=config.LoRA_dropout,
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True, padding='max_length', max_length=reward_config.max_length, truncation=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=1)
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    def preprocess_func(examples):
        inputs_chosen, inputs_rejected = [], []
        attention_masks_chosen, attention_masks_rejected = [], []
        margins = []
        for i in range(len(examples['chosen'])):
            logits_chosen = float(examples['logits_chosen'][i])
            logits_rejected = float(examples['logits_rejected'][i])
            if logits_chosen > logits_rejected:
                tokenized_our_choice = tokenizer(examples['chosen'][i], truncation="longest_first", padding='longest', max_length=reward_config.max_length)
                tokenized_our_rejected = tokenizer(examples['rejected'][i], truncation="longest_first", padding='longest', max_length=reward_config.max_length)
                margin = logits_chosen - logits_rejected
            else:
                tokenized_our_choice = tokenizer(examples['rejected'][i], truncation="longest_first", padding='longest', max_length=reward_config.max_length)
                tokenized_our_rejected = tokenizer(examples['chosen'][i], truncation="longest_first", padding='longest', max_length=reward_config.max_length)
                margin = logits_rejected - logits_chosen

            inputs_chosen.append(tokenized_our_choice['input_ids'])
            attention_masks_chosen.append(tokenized_our_choice['attention_mask'])
            inputs_rejected.append(tokenized_our_rejected['input_ids'])
            attention_masks_rejected.append(tokenized_our_rejected['attention_mask'])
            margins.append(margin)

        d = {
            'input_ids_chosen': inputs_chosen,
            'attention_mask_chosen': attention_masks_chosen,
            'input_ids_rejected': inputs_rejected,
            'attention_mask_rejected': attention_masks_rejected
        }
        if config.margin == "True":
            d['margin'] = margins
        return d

    # preprocess the dataset
    train_dataset = train_dataset.map(
        preprocess_func,
        batched=True,
        num_proc=config.num_proc,
    )
    test_dataset = test_dataset.map(
        preprocess_func,
        batched=True,
        num_proc=config.num_proc,
    )
    print("begin training")
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset['train'],
        eval_dataset=test_dataset['train'],
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(reward_config.output_dir)
    tokenizer.save_pretrained(reward_config.output_dir)
