import json
import math
import os
import sys

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Run with
""" 
for num_run in {1..3}; do for num_perspective in {1..3..2}; do /nas/ucb/constantinweisser/anaconda3/envs/llmperspective/bin/python /nas/ucb/constantinweisser/First-third-person-feedback-experiments/src/eval_pm/eval.py $num_perspectiv
e $num_run; done; done
"""

os.environ["HF_HOME"] = "/nas/ucb/marcuswilliams/cache/"

# Paths
perspective = f"3_{sys.argv[1]}"
# perspective = "untrained"
# PM_path = f"models/fair_{perspective}"
# model_name = "sfairXC/FsfairX-LLaMA3-RM-v0.1"
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "Ray2333/GRM-llama3-8B-sftreg"
tokenizer_name = "sfairXC/FsfairX-LLaMA3-RM-v0.1"
# PM_path = f"models/llama_g3_{perspective}/checkpoint-500"
model_specifier = f"_trainonperso-intended_grm_big_1epochs_lr5e-4_{sys.argv[2]}"
# model_specifier = f"_hh_1epochs_lr5e-4_{sys.argv[2]}"
PM_path = f"models/llama_{perspective}{model_specifier}"

print("perspective ", perspective)
print("model_specifier ", model_specifier)


# Load the dataset
num_proc = 4


# Set device
device = "cuda:1"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# preference_model = AutoModelForSequenceClassification.from_pretrained(PM_path, num_labels=1)
preference_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# Load the LoRA configuration and initialize the PEFT model
# lora_config = LoraConfig.from_pretrained(PM_path)
TEST_ON_HALF = True

# #preference_model = get_peft_model(preference_model, lora_config)
if perspective != "untrained":
    preference_model.load_adapter(PM_path, "a")
    preference_model.enable_adapters()
    print(preference_model.active_adapters())
# print(preference_model.get_model_status())
# Ensure the model is in evaluation mode and on the correct device
preference_model.eval().to(device)

# Preprocess and tokenize the input for the model
with torch.no_grad():
    for task in ["sycophancy", "danger_refusal", "impossible_task_refusal", "verbosity", "personalisation"]:
        dataset_path = f"data/datasets/{task}.jsonl"
        save_path = f"data/datasets/{task}_{perspective}_PM_eval{model_specifier}.jsonl"
        dataset = load_dataset("json", data_files=dataset_path)["train"]

        if TEST_ON_HALF:
            len_df = len(dataset)
            dataset = dataset.select(list(range(len_df // 2, len_df)))

        positive_scores = []
        negative_scores = []
        for item in tqdm(dataset):
            # print(item)

            positive_label = item["positive_label"]
            negative_label = item["negative_label"]
            pos_messages = [
                {"role": "user", "content": item["prompt"] + " " + item["question"]},
                {"role": "assistant", "content": item[positive_label]},
            ]
            neg_messages = [
                {"role": "user", "content": item["prompt"] + " " + item["question"]},
                {"role": "assistant", "content": item[negative_label]},
            ]
            pos_formated = tokenizer.apply_chat_template(
                pos_messages, tokenize=False, add_generation_prompt=False
            ).replace(tokenizer.bos_token, "")
            neg_formated = tokenizer.apply_chat_template(
                neg_messages, tokenize=False, add_generation_prompt=False
            ).replace(tokenizer.bos_token, "")
            pos_tokenized = tokenizer(pos_formated, max_length=2048, truncation=True, return_tensors="pt").to(device)
            pos_output = preference_model(**pos_tokenized)

            pos_logits = pos_output.logits
            positive_scores.append(pos_logits.item())

            neg_tokenized = tokenizer(neg_formated, max_length=512, truncation=True, return_tensors="pt").to(device)
            neg_output = preference_model(**neg_tokenized)
            neg_logits = neg_output.logits
            negative_scores.append(neg_logits.item())

        with open(save_path, "w") as f:
            print(f"Saving to {save_path}")
            for i, item in enumerate(dataset):
                positive_label = item["positive_label"]
                negative_label = item["negative_label"]
                item[positive_label + "_score"] = positive_scores[i]
                item[negative_label + "_score"] = negative_scores[i]
                f.write(json.dumps(item) + "\n")
