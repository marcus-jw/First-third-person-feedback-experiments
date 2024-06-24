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

# Load the dataset
num_proc = 4
test_dataset = load_dataset('json', data_files=f'data/hh_labels/hh_test_1_1.jsonl') 


# Set device
device = "cuda"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(PM_path)
preference_model = AutoModelForSequenceClassification.from_pretrained(PM_path, num_labels=1)

# Load the LoRA configuration and initialize the PEFT model
lora_config = LoraConfig.from_pretrained(PM_path)
#print("Loaded LoraConfig:")
#print(lora_config)

preference_model = get_peft_model(preference_model, lora_config)

# Ensure the model is in evaluation mode and on the correct device
preference_model.eval().to(device)

# Preprocess and tokenize the input for the model
with torch.no_grad():
    for item in tqdm(dataset):
        chosen_tokenized = tokenizer(item["chosen"] , max_length=512, truncation=True, return_tensors="pt").to(device)
        chosen_output = preference_model(**tokenized)
        chosen_logits = output.logits

        rejected_tokenized = tokenizer(item["rejected"] , max_length=512, truncation=True, return_tensors="pt").to(device)
        rejected_output = preference_model(**rejected_tokenized)
        rejected_logits = rejected_output.logits

        if chosen_logits.item() > rejected_logits.item():
            correct += 1
print(f"Accuracy: {correct/len(dataset)}")

        