import os
import random
import json
from tqdm import tqdm
import time
import argparse
from datasets import load_dataset
import re
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic()

feedback_model = "claude-3-haiku-20240307"
train_test = "train"

# "harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"
split = "helpful-rejection-sampled"
start_id = 86737

dataset = load_dataset("polinaeterna/hh-rlhf", split)

save_path = os.path.dirname(__file__) + f"/../../data/hh_labels/haiku_{split}_{train_test}_requests"

first_firstshoes_system_prompt = ("You are a human user testing a new chatbot. Your job is to have a conversation with the chatbot. " 
"For some responses the chatbot will provide two responses and you should rate which one you, the human user, think is better.")

third_firstshoes_system_prompt = ("You will be given a conversation between a human user and a chatbot. For the final response, the chatbot will provide two responses, "
                              "it is your job to rate which out of the two responses the human user would prefer.")

third_thirdshoes_system_prompt = ("You will be given a conversation between a human user and a chatbot. For the final response, the chatbot will provide two responses, "
                                "it is your job to rate which out of the two responses is better.")

first_thirdshoes_system_prompt = ("You are a human user testing a new chatbot. Your job is to have a conversation with the chatbot. " 
"For some responses the chatbot will provide two responses and you should rate which one an independent third-party observer would think is better.")
def prepare_request(system_prompt, messages, metadata):
    request = {
        "model": feedback_model,
        "max_tokens": 1,
        "temperature": 0,
        "system": system_prompt,
        "messages": messages,
        "metadata": metadata
    }
    return request

def first_firstshoes_person(d: dict):
    messages = []
    messages.append({"role": "user", "content": "Hello I am a chatbot, what would you like to talk about?"})
    for i in d["conversation"]:
        if i["role"] == "prompter":
            messages.append({"role": "assistant", "content": i['content'].strip()})
        else:
            messages.append({"role": "user", "content": i['content'].strip()})
    messages.append({"role": "user", "content": f"The chatbot will now provide two responses. Please rate which option you, the user, would prefer. Answer only with A or B.\n\nResponse A: {d['responseA']}\n\nResponse B: {d['responseB']}"})
    messages.append({"role": "assistant", "content": "I would prefer option:"})
    metadata = {"id": d["id"], "chosen": d["chosen"]}    
    metadata = {"id": d["id"], "chosen": d["chosen"]}
    return messages, metadata

def third_firstshoes_person(d: dict):
    conversation = ""
    for i in d["conversation"]:
        if i["role"] == "prompter":
            conversation += f"Human: {i['content'].strip()}\n"
        else:
            conversation += f"Chatbot: {i['content'].strip()}\n"
    messages = [{"role": "user", "content": f"The conversation between the chatbot and the human is:\n\n{conversation}\n\nThe two response options are:\n\nResponse A: {d['responseA']}\n\n Response B: {d['responseB']}\n\nPlease choose the option the human would prefer. Answer only with A or B."},
                {"role": "assistant", "content":"The option the human would prefer is option:"}]
    metadata = {"id": d["id"], "chosen": d["chosen"]}
    return messages, metadata

def first_thirdshoes_person(d: dict):
    messages = []
    messages.append({"role": "user", "content": "Hello I am a chatbot, what would you like to talk about?"})
    for i in d["conversation"]:
        if i["role"] == "prompter":
            messages.append({"role": "assistant", "content": i['content'].strip()})
        else:
            messages.append({"role": "user", "content": i['content'].strip()})
    
    messages.append({"role": "user", "content": f"The chatbot will now provide two responses. Please rate which option an independent third-party observer would prefer. Answer only with A or B.\n\nResponse A: {d['responseA']}\n\nResponse B: {d['responseB']}"})
    messages.append({"role": "assistant", "content": "I think an independent third-party observer would prefer option:"})
    metadata = {"id": d["id"], "chosen": d["chosen"]}
    return messages, metadata

def third_thirdshoes_person(d: dict):
    conversation = ""
    for i in d["conversation"]:
        if i["role"] == "prompter":
            conversation += f"Human: {i['content'].strip()}\n"
        else:
            conversation += f"Chatbot: {i['content'].strip()}\n"
    messages = [{"role": "user", "content": f"The conversation between the chatbot and the human is:\n\n{conversation}\n\nThe two response options are:\n\nResponse A: {d['responseA']}\n\n Response B: {d['responseB']}\n\nPlease chose the option an independent third-party observer would prefer. Answer only with A or B."},
                {"role": "assistant", "content":"The option an independent third-party observer would prefer is:"}]
    metadata = {"id": d["id"], "chosen": d["chosen"]}
    return messages, metadata

if train_test == "train":
    dataset = dataset["train"]
else:
    dataset = dataset["test"]

pattern = r'\n\nAssistant:|\n\nHuman:' 
with open(save_path+"_1_1.jsonl", "w", encoding="utf-8") as f11, open(save_path+"_3_1.jsonl", "w", encoding="utf-8") as f31, open(save_path+"_1_3.jsonl", "w", encoding="utf-8") as f13, open(save_path+"_3_3.jsonl", "w", encoding="utf-8") as f33:
    print(dataset[36])
    for i, line in enumerate(dataset):
        chosen = line["chosen"]
        rejected = line["rejected"]
        conversation = re.split(pattern, line["chosen"])[1:]
        chosen_answer = conversation[-1]
        conversation = conversation[:-1]
        rejected_answer = re.split(pattern, line["rejected"])[-1]
        messages = []
        conversation = [e.strip() for e in conversation]
        if "" in conversation: # anthropic dataset has some errors
            conversation.remove("")
        for j, e in enumerate(conversation):
            if j % 2 == 0:
                messages.append({"role": "prompter", "content": e.strip()})
            else:
                messages.append({"role": "answerer", "content": e.strip()})
        chosen = random.choice(["A", "B"])
        A = chosen_answer if chosen == "A" else rejected_answer
        B = rejected_answer if chosen == "A" else chosen_answer
        d = {"conversation": messages, "responseA": A, "responseB": B, "chosen": chosen, "id": i+start_id}
        
        messages, metadata = first_firstshoes_person(d)
        request = prepare_request(first_firstshoes_system_prompt, messages, metadata)
        f11.write(json.dumps(request))
        f11.write("\n")
        
        messages, metadata = third_firstshoes_person(d)
        request = prepare_request(third_firstshoes_system_prompt, messages, metadata)
        f31.write(json.dumps(request))
        f31.write("\n")
        
        messages, metadata = first_thirdshoes_person(d)
        request = prepare_request(first_thirdshoes_system_prompt, messages, metadata)
        f13.write(json.dumps(request))
        f13.write("\n")
        
        messages, metadata = third_thirdshoes_person(d)
        request = prepare_request(third_thirdshoes_system_prompt, messages, metadata)
        f33.write(json.dumps(request))
        f33.write("\n")