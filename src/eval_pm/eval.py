import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from tqdm import tqdm
from openai import OpenAI
import math

client = OpenAI()

# Paths
PM_path = "models/Llama/"
task = "sycohpancy"
dataset_path = f"data/datasets/{task}.jsonl"
save_path = f"data/datasets/{task}_PM_eval.jsonl"

# Load the dataset
num_proc = 4
dataset = load_dataset('json', data_files=dataset_path)


# Set device
device = "cuda"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(PM_path)
preference_model = AutoModelForSequenceClassification.from_pretrained(PM_path, num_labels=1)

# Load the LoRA configuration and initialize the PEFT model
lora_config = LoraConfig.from_pretrained(PM_path)
print("Loaded LoraConfig:")
print(lora_config)

preference_model = get_peft_model(preference_model, lora_config)

# Ensure the model is in evaluation mode and on the correct device
preference_model.eval().to(device)

# Preprocess and tokenize the input for the model
with torch.no_grad():
    for item in tqdm(dataset):
        positive_label = item['positive_label']
        negative_label = item['negative_label']
        positive_scores = []
        negative_scores = []

        pos_tokenized = tokenizer(f"Human: {item['prompt']} {item['question']}\nAssistant: {item[positive_label]}" , max_length=512, truncation=True, return_tensors="pt").to(device)
        pos_output = preference_model(**tokenized)


        pos_logits = output.logits
        positive_scores.append(pos_logits.item())

        neg_tokenized = tokenizer(f"Human: {item['prompt']} {item['question']}\nAssistant: {item[negative_label]}" , max_length=512, truncation=True, return_tensors="pt").to(device)
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

