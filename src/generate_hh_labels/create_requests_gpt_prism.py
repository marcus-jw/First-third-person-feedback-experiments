
import os
import random
import json
from tqdm import tqdm
import time
import argparse
from datasets import load_dataset
import re
from datasets import load_dataset
import numpy as np
import pandas as pd
import itertools
from copy import deepcopy



feedback_model = "gpt-3.5-turbo"
train_test = "train"


save_path = f"data/hh_labels/prism_35_{train_test}_requests"
first_firstshoes_system_prompt = ("You are a human user testing a new chatbot. Your job is to have a conversation with the chatbot. " 
"For some responses the chatbot will provide two responses and you should rate which one you, the human user, think is better.")

third_firstshoes_system_prompt = ("You will be given a conversation between a human user and a chatbot. For the final response, the chatbot will provide two responses, "
                              "it is your job to rate which out of the two responses the human user would prefer. Answer only with A or B.")

third_thirdshoes_system_prompt = ("You will be given a conversation between a human user and a chatbot. For the final response, the chatbot will provide two responses, "
                                "it is your job to rate which out of the two responses an independent third-party observer would prefer. Answer only with A or B.")

first_thirdshoes_system_prompt = ("You are a human user testing a new chatbot. Your job is to have a conversation with the chatbot. " 
"For some responses the chatbot will provide two responses and you should rate which one an independent third-party observer would think is better.")

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
        if i["role"] == "user":
            messages.append({"role": "assistant", "content": i['content'].strip()})
        else:
            messages.append({"role": "user", "content": i['content'].strip()})
    messages.append({"role": "system", "content": "The chatbot will now provide two responses. Please rate which one is better. Answer only with A or B."})
    
    messages.append({"role": "user", "content": f"Response A: {d['responseA']}\n\n Response B: {d['responseB']}" })
    metadata = {"id": d["id"], "chosen": d["chosen"], "scoreA": d["scoreA"], "scoreB": d["scoreB"]}
    return messages, metadata
def third_firstshoes_person(d:dict, CoT = False):
    messages = [{"role": "system", "content": third_firstshoes_system_prompt}]
    conversation = ""
    for i in d["conversation"]:
        if i["role"] == "user":
            conversation += f"Human: {i['content'].strip()}\n"
        else:
            conversation += f"Chatbot: {i['content'].strip()}\n"
    messages.append({"role": "user", "content": f"The conversation between the chatbot and the human is:\n\n<conversation>\n\n{conversation}\n</conversation>\n\nThe two response options are:\n\nResponse A: {d['responseA']}\n\n Response B: {d['responseB']}\n\nPlease choose the option the human would prefer. Answer only with A or B." })
    metadata = {"id": d["id"], "chosen": d["chosen"], "scoreA": d["scoreA"], "scoreB": d["scoreB"]}
    return messages, metadata
def first_thirdshoes_person(d:dict, CoT = False):
    messages = [{"role": "system", "content": first_thirdshoes_system_prompt}]
    for i in d["conversation"]:
        if i["role"] == "user":
            messages.append({"role": "assistant", "content": i['content'].strip()})
        else:
            messages.append({"role": "user", "content": i['content'].strip()})
    messages.append({"role": "system", "content": "The chatbot will now provide two responses. Please rate which a outside observer would prefer. Answer only with A or B."})
    
    messages.append({"role": "user", "content": f"Response A: {d['responseA']}\n\n Response B: {d['responseB']}" })
    metadata = {"id": d["id"], "chosen": d["chosen"], "scoreA": d["scoreA"], "scoreB": d["scoreB"]}
    return messages, metadata
def third_thirdshoes_person(d:dict, CoT = False):
    messages = [{"role": "system", "content": third_thirdshoes_system_prompt}]
    conversation = ""
    for i in d["conversation"]:
        if i["role"] == "user":
            conversation += f"Human: {i['content'].strip()}\n"
        else:
            conversation += f"Chatbot: {i['content'].strip()}\n"
    messages.append({"role": "user", "content": f"The conversation between the chatbot and the human is:\n\n<conversation>\n\n{conversation}\n</conversation>\n\nThe two response options are:\n\nResponse A: {d['responseA']}\n\n Response B: {d['responseB']}\n\nPlease choose the an independent third party observer would prefer. Answer only with A or B." })
    metadata = {"id": d["id"], "chosen": d["chosen"], "scoreA": d["scoreA"], "scoreB": d["scoreB"]}
    return messages, metadata


def generate_comparison_from_conversation_history(conversation_history, i_request=0):
    i_turn = 0
    i_message = 0

    requests = [] #{"conversation": messages, "responseA": A, "responseB": B, "chosen": chosen, "id": i+start_id}
    conversation = []

    while True:
        #print("Running on message ", i_message, len(requests))
        # get options
        if i_message>len(conversation_history)-1:
            #print("We ran over all messages")
            break
        user_message = conversation_history[i_message]
        assert user_message["role"]=="user", f'Role should be user, but is {user_message["role"]}'
        assert user_message["turn"]==i_turn, f'Turn should be {i_turn}, but is {user_message["turn"]}'
        conversation.append({"role": "user", "content": user_message["content"]})
        i_message +=1 

        
        model_message_chosen = None
        model_messages = []
        while True:
            if i_message>len(conversation_history)-1:
                break
            model_message = conversation_history[i_message]
            if model_message["role"]!="model":
                break
            else:
                assert model_message["turn"]==i_turn, f'Turn should be {i_turn}, but is {user_message["turn"]}'
                if model_message["if_chosen"]: 
                    model_message_chosen = model_message
                model_messages.append(model_message)
                i_message +=1

        random.shuffle(model_messages)
        #print("model_messages ", model_messages)
        for msg_A, msg_B in itertools.combinations(model_messages, 2):
            chosen = "A" if msg_A["score"]> msg_B["score"] else "B"
            requests.append({"conversation": deepcopy(conversation), "responseA": msg_A["content"], "responseB": msg_B["content"], "scoreA": msg_A["score"], "scoreB": msg_B["score"], "chosen": chosen, "id": i_request})
            i_request += 1
        
        conversation.append({"role": "assistant", "content": model_message_chosen["content"]})    

        #print("user ", user_message["content"])
        #print("model ", [msg["content"] for msg in model_messages])
        i_turn += 1


    return requests


if __name__=="__main__":
    dataset = load_dataset("HannahRoseKirk/prism-alignment", "conversations")
    dataset = dataset["train"]
    df = pd.DataFrame(data=dataset)
    df = df[:2]

    with open(save_path+"_3_1.jsonl", "w",encoding="utf-8") as f31, open(save_path+"_3_3.jsonl", "w",encoding="utf-8") as f33, open(save_path+"_all.jsonl", "w",encoding="utf-8") as fall:
        comparisons_all = []
        for i_conversation, conversation_history in enumerate(df["conversation_history"]):
            if i_conversation%100==0: print("Running on conversation ", i_conversation)
            comparisons = generate_comparison_from_conversation_history(conversation_history, i_request=len(comparisons_all))
            comparisons_all.extend(comparisons)
        
        print("comparisons_all ", len(comparisons_all))
        for comparison in comparisons_all:
            fall.write(json.dumps(comparison))
            fall.write("\n")

            
            messages, metadata = third_firstshoes_person(comparison)
            request = prepare_request(feedback_model,messages,metadata)
            f31.write(json.dumps(request))
            f31.write("\n")

            messages, metadata = third_thirdshoes_person(comparison)
            request = prepare_request(feedback_model,messages,metadata)
            f33.write(json.dumps(request))
            f33.write("\n")

