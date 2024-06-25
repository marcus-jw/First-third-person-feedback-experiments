import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from tqdm import tqdm
import math

# Paths
PM_path = "models/final/"
model_name = "meta-llama/Meta-Llama-3-8B"
# Load the dataset
num_proc = 4
dataset = load_dataset('json', data_files=f'data/hh_labels/hh_test_3_1.jsonl')['train']


# Set device
device = "cuda:7"
with torch.no_grad():

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preference_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).eval()

    # Load the LoRA configuration and initialize the PEFT model
    print("Loading LoraConfig:")
    lora_config = LoraConfig.from_pretrained(PM_path)
    
    #print(lora_config)

    preference_model = get_peft_model(preference_model, lora_config)

    # Ensure the model is in evaluation mode and on the correct device
    preference_model.eval().to(device)
    correct = 0
    # Preprocess and tokenize the input for the model
    for item in tqdm(dataset):
        chosen_tokenized = tokenizer(item["chosen"] , max_length=512, truncation=True, return_tensors="pt").to(device)
        chosen_output = preference_model(**chosen_tokenized)
        chosen_logits = chosen_output.logits

        rejected_tokenized = tokenizer(item["rejected"] , max_length=512, truncation=True, return_tensors="pt").to(device)
        rejected_output = preference_model(**rejected_tokenized)
        rejected_logits = rejected_output.logits

        if chosen_logits.item() > rejected_logits.item():
            correct += 1
    print(f"Accuracy: {correct/len(dataset)}")

            