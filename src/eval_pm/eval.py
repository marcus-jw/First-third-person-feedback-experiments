import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import math
import os
os.environ["HF_HOME"] = "/nas/ucb/marcuswilliams/cache/"

# Paths
perspective = "none"
PM_path = f"models/fair_{perspective}"
model_name = "sfairXC/FsfairX-LLaMA3-RM-v0.1"


# Load the dataset
num_proc = 4


# Set device
device = "cuda:1"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
#preference_model = AutoModelForSequenceClassification.from_pretrained(PM_path, num_labels=1)
preference_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# Load the LoRA configuration and initialize the PEFT model
# lora_config = LoraConfig.from_pretrained(PM_path)


# #preference_model = get_peft_model(preference_model, lora_config)
# preference_model.load_adapter(PM_path,"a")
# preference_model.enable_adapters()
#print(preference_model.get_model_status())
# Ensure the model is in evaluation mode and on the correct device
preference_model.eval().to(device)

# Preprocess and tokenize the input for the model
with torch.no_grad():
    for task in ["sycophancy","danger_refusal","impossible_task_refusal","verbosity","personalisation"]:
        dataset_path = f"data/datasets/{task}.jsonl"
        save_path = f"data/datasets/{task}_{perspective}_PM_eval.jsonl"
        dataset = load_dataset('json', data_files=dataset_path)["train"]

        positive_scores = []
        negative_scores = []
        for item in tqdm(dataset):
            #print(item)

            positive_label = item['positive_label']
            negative_label = item['negative_label']
            pos_messages = [{"role": "user", "content": item["prompt"] + " " + item['question']}, {"role": "assistant", "content": item[positive_label]}]
            neg_messages = [{"role": "user", "content": item["prompt"] + " " + item['question']}, {"role": "assistant", "content": item[negative_label]}]
            pos_formated = tokenizer.apply_chat_template(pos_messages, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
            neg_formated = tokenizer.apply_chat_template(neg_messages, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
            pos_tokenized = tokenizer(pos_formated , max_length=2048, truncation=True, return_tensors="pt").to(device)
            pos_output = preference_model(**pos_tokenized)


            pos_logits = pos_output.logits
            positive_scores.append(pos_logits.item())

            neg_tokenized = tokenizer(neg_formated , max_length=512, truncation=True, return_tensors="pt").to(device)
            neg_output = preference_model(**neg_tokenized)
            neg_logits = neg_output.logits
            negative_scores.append(neg_logits.item())

            
        with open(save_path, 'w') as f:
            for i,item in enumerate(dataset):
                positive_label = item['positive_label']
                negative_label = item['negative_label']
                item[positive_label + "_score"] = positive_scores[i]
                item[negative_label + "_score"] = negative_scores[i]
                f.write(json.dumps(item) + '\n')

