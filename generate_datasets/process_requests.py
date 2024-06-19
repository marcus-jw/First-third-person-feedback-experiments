from openai import OpenAI
import os
import random
import json
from tqdm import tqdm
import time
client = OpenAI()
train_test = "test"
for perspective in ["1_1","3_1","1_3","3_3"]:
    response_path = f"generate_datasets/{train_test}_results_{perspective}.jsonl"
    save_path = f"datasets/hh_{train_test}_{perspective}.jsonl"
    with open(response_path, 'r', encoding='utf-8') as infile, open(save_path, 'w', encoding='utf-8') as outfile:

        responses = infile.readlines()
        responses = [json.loads(response.strip()) for response in responses]
        # sort the responses by id
        responses = sorted(responses, key=lambda x: x[2]["id"])
        for i in range(len(responses)):
            logprob_A = -10
            logprob_B = -10
            for token in responses[i][1]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
                if token["token"] == "A":
                    logprob_A = token["logprob"]
                if token["token"] == "B":
                    logprob_B = token["logprob"]
            logprob_chosen = logprob_A if responses[i][2]["chosen"] == "A" else logprob_B
            logprob_rejected = logprob_B if responses[i][2]["chosen"] == "A" else logprob_A
            outfile.write(json.dumps({"logits_chosen":logprob_chosen,"logits_rejected":logprob_rejected}) + '\n')

