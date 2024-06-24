from openai import OpenAI
import os
import json
from datasets import load_dataset
from tqdm import tqdm

train_test = "train"
split = "harmless-base"
#"harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"
dataset = load_dataset("polinaeterna/hh-rlhf", split)    
if train_test == "train":
    dataset = dataset["train"]
else:
    dataset = dataset["test"]
#dataset = dataset.select(range(100)) # For testing purposes

perspectives = ["1_1", "3_1", "1_3", "3_3"]

# Pre-load all response files
responses = {}
for perspective in perspectives:
    response_path = os.path.dirname(__file__) + f"/../../data/hh_labels/haiku_{split}_{train_test}_results_{perspective}.jsonl"
    with open(response_path, 'r', encoding='utf-8') as infile:
        responses[perspective] = [json.loads(line.strip()) for line in infile]
        responses[perspective] = sorted(responses[perspective], key=lambda x: x[2]["id"])

# Prepare output files
outfiles = {}
for perspective in perspectives:
    save_path = os.path.dirname(__file__) + f"/../../data/hh_labels/haiku_{split}_{train_test}_{perspective}.jsonl"
    outfiles[perspective] = open(save_path, 'w', encoding='utf-8')

# Process datapoints
for i in tqdm(range(len(dataset)), desc="Processing datapoints"):
    chosen = dataset[i]["chosen"]
    rejected = dataset[i]["rejected"]
    
    datapoint = {
        "chosen": chosen,
        "rejected": rejected,
        "prob_chosen": {},
        "prob_rejected": {}
    }
    
    valid_datapoint = True
    
    for perspective in perspectives:
        try:
            prob_chosen = 1 if responses[perspective][i][2]["chosen"] in responses[perspective][i][1]["content"][0]["text"] else 0
            prob_rejected = 1 - prob_chosen
            
            datapoint["prob_chosen"][perspective] = prob_chosen
            datapoint["prob_rejected"][perspective] = prob_rejected
        except:
            valid_datapoint = False
            break
    
    if valid_datapoint:
        for perspective in perspectives:
            output_datapoint = {
                "chosen": datapoint["chosen"],
                "rejected": datapoint["rejected"],
                "prob_chosen": datapoint["prob_chosen"][perspective],
                "prob_rejected": datapoint["prob_rejected"][perspective]
            }
            outfiles[perspective].write(json.dumps(output_datapoint) + '\n')

# Close all output files
for outfile in outfiles.values():
    outfile.close()