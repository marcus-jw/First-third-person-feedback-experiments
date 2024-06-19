from openai import OpenAI
import os
import random
import json
from tqdm import tqdm
import time
import argparse
from datasets import load_dataset
import re
client = OpenAI()

feedback_model = "gpt-3.5-turbo-0125"
train_test = "test"

save_path = f"API/{train_test}_requests"
first_firstshoes_system_prompt= "Your job is to have a conversation with a chatbot as if you're a human, and then rate which out of two chatbot responses is better when prompted to do so. "

third_firstshoes_system_prompt = ("You will be given a conversation between a human and a chatbot. For the final response, the chatbot will provide two responses, "
                              "it is your job to rate which out of the two responses the human would prefer. ")

third_thirdshoes_system_prompt = ("You will be given a conversation between a human and a chatbot. For the final response, the chatbot will provide two responses, "
                                "it is your job to rate which out of the two responses is better. ")

first_thirdshoes_system_prompt = "Your job is to have a conversation with a chatbot as if you're a human, and then rate which out of two chatbot responses a outside observer would prefer."

def prepare_request(model,messages,metadata):
    request = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 5,
        "metadata": metadata
    }
    return request
    
def first_firstshoes_person(d:dict, CoT = False):
    messages = [{"role": "system", "content": first_firstshoes_system_prompt}]
    for i in d["conversation"]:
        if i["role"] == "prompter":
            messages.append({"role": "assistant", "content": i['content']})
        else:
            messages.append({"role": "user", "content": i['content']})
    messages.append({"role": "system", "content": "The chatbot will now provide two responses. Please rate which one is better. Answer only with A or B."})
    
    messages.append({"role": "user", "content": f"Response A: {d['responseA']}\n\n Response B: {d['responseB']}" })
    metadata = {"id": d["id"], "chosen": d["chosen"]}
    return messages, metadata
def third_firstshoes_person(d:dict, CoT = False):
    messages = [{"role": "system", "content": third_firstshoes_system_prompt}]
    conversation = ""
    for i in d["conversation"]:
        if i["role"] == "prompter":
            conversation += f"Human: {i['content']}"
        else:
            conversation += f"Chatbot: {i['content']}"
    messages.append({"role": "user", "content": f"The conversation between the chatbot and the human is {conversation}\n\nThe two response options are:\n\nResponse A: {d['responseA']}\n\n Response B: {d['responseB']}\n\nPlease chose the option the human would prefer. Answer only with A or B." })
    metadata = {"id": d["id"], "chosen": d["chosen"]}
    return messages, metadata
def first_thirdshoes_person(d:dict, CoT = False):
    messages = [{"role": "system", "content": first_thirdshoes_system_prompt}]
    for i in d["conversation"]:
        if i["role"] == "prompter":
            messages.append({"role": "assistant", "content": i['content']})
        else:
            messages.append({"role": "user", "content": i['content']})
    messages.append({"role": "system", "content": "The chatbot will now provide two responses. Please rate which a outside observer would prefer. Answer only with A or B."})
    
    messages.append({"role": "user", "content": f"Response A: {d['responseA']}\n\n Response B: {d['responseB']}" })
    metadata = {"id": d["id"], "chosen": d["chosen"]}
    return messages, metadata
def third_thirdshoes_person(d:dict, CoT = False):
    messages = [{"role": "system", "content": third_thirdshoes_system_prompt}]
    conversation = ""
    for i in d["conversation"]:
        if i["role"] == "prompter":
            conversation += f"Human: {i['content']}"
        else:
            conversation += f"Chatbot: {i['content']}"
    messages.append({"role": "user", "content": f"The conversation between the chatbot and the human is {conversation}\n\nThe two response options are:\n\nResponse A: {d['responseA']}\n\n Response B: {d['responseB']}\n\nPlease chose the option the human would prefer. Answer only with A or B." })
    metadata = {"id": d["id"], "chosen": d["chosen"]}
    return messages, metadata

    
dataset = load_dataset("Anthropic/hh-rlhf")     
if train_test == "train":
    dataset = dataset["train"]
else:
    dataset = dataset["test"]

dataset = dataset.select(range(5)) # For testing purposes


pattern = r'\n\nAssistant:|\n\nHuman:'
with open(save_path+"_1_1.jsonl", "w",encoding="utf-8") as f11, open(save_path+"_3_1.jsonl", "w",encoding="utf-8") as f31, open(save_path+"_1_3.jsonl", "w",encoding="utf-8") as f13, open(save_path+"_3_3.jsonl", "w",encoding="utf-8") as f33:
    for i,line in enumerate(dataset):
        chosen = line["chosen"]
        rejected = line["rejected"]
        conversation = re.split(pattern, line["chosen"])[1:]
        chosen_answer = conversation[-1]
        conversation = conversation[:-1]
        rejected_answer = re.split(pattern, line["rejected"])[-1]
        messages = []
        for i, e in enumerate(conversation):
            if i % 2 == 0:
                messages.append({"role": "prompter", "content": e.strip()})
            else:
                messages.append({"role": "answerer", "content": e.strip()})
        chosen = random.choice(["A", "B"])
        A = chosen_answer if chosen == "A" else rejected_answer
        B = rejected_answer if chosen == "A" else chosen_answer
        d = {"conversation": messages, "responseA": A, "responseB": B, "chosen": chosen, "id": i}
        
        messages, metadata = first_firstshoes_person(d)
        request = prepare_request(feedback_model,messages,metadata)
        f11.write(json.dumps(request))
        f11.write("\n")
        messages, metadata = third_firstshoes_person(d)
        request = prepare_request(feedback_model,messages,metadata)
        f31.write(json.dumps(request))
        f31.write("\n")
        messages, metadata = first_thirdshoes_person(d)
        request = prepare_request(feedback_model,messages,metadata)
        f13.write(json.dumps(request))
        f13.write("\n")
        messages, metadata = third_thirdshoes_person(d)
        request = prepare_request(feedback_model,messages,metadata)
        f33.write(json.dumps(request))
        f33.write("\n")
