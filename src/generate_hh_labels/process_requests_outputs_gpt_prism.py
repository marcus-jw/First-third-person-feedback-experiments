from openai import OpenAI
import os
import random
import json
from tqdm import tqdm
import time
from datasets import load_dataset
train_test = "train"
#split = "harmless-base"
#"harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"
#dataset = load_dataset("polinaeterna/hh-rlhf", split)     
# if train_test == "train":
#     dataset = dataset["train"]
# else:
#     dataset = dataset["test"]
#dataset = dataset.select(range(1000)) # For testing purposes
#dataset = load_dataset("Anthropic/hh-rlhf")["test"]



for perspective in ["3_1","3_3"]:
    response_path = f"data/hh_labels/prism_35_{train_test}_results_{perspective}.jsonl"
    comparison_path = f"data/hh_labels/prism_35_{train_test}_requests_all.jsonl"
    save_path = f"data/hh_labels/prism_35_{train_test}_{perspective}.jsonl"
    
    with open(response_path, 'r', encoding='utf-8') as responsefile, open(comparison_path, 'r', encoding='utf-8') as comparisonfile, open(save_path, 'w', encoding='utf-8') as outfile:

        responses = responsefile.readlines()
        responses = [json.loads(response.strip()) for response in responses]
        # sort the responses by id
        responses = sorted(responses, key=lambda x: x[2]["id"])

        comparisons = comparisonfile.readlines()
        comparisons = [json.loads(comparison.strip()) for comparison in comparisons]
        assert len(responses) == len(comparisons), f"Number of responses is {len(responses)} and number of comparisons is {len(comparisons)}. they should be the same"

        
        for i in range(len(responses)):


            answer_chosen = comparisons[i]["responseA"] if comparisons[i]["chosen"]=="A" else comparisons[i]["responseB"]
            answer_rejected = comparisons[i]["responseB"] if comparisons[i]["chosen"]=="A" else comparisons[i]["responseA"]


            #chosen=dataset[i]["chosen"]
            #rejected=dataset[i]["rejected"]
            logprob_A = -10
            logprob_B = -10
            print(i)
            for token in responses[i][1]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
                if token["token"] == "A":
                    logprob_A = token["logprob"]
                if token["token"] == "B":
                    logprob_B = token["logprob"]
            logprob_chosen = logprob_A if responses[i][2]["chosen"] == "A" else logprob_B
            logprob_rejected = logprob_B if responses[i][2]["chosen"] == "A" else logprob_A
            
            outfile.write(json.dumps({
                "conversation": comparisons[i]["conversation"],
                "answer_chosen": answer_chosen,
                "answer_rejected": answer_rejected,
                "logits_chosen":logprob_chosen,
                "logits_rejected":logprob_rejected
                }) + '\n')

