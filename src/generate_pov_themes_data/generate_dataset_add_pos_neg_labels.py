# Only used because initially we forgot to add the entries for positive and negative labels
# Because this has now been incorporated, this file should be deprecated


from openai import OpenAI
import json
from tqdm import tqdm
from itertools import cycle


client = OpenAI()

model = "gpt-4o"
task_list = ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']

pos_neg_key_label_dict = {
    "danger_refusal": ["refusal_answer", "dangerous_answer"],
    "impossible_task_refusal": ["refusal_answer", "incorrect_answer"],
    "personalisation": ["appropriate_answer", "inappropriate_answer"],
    "sycophancy": ["sycophantic_answer", "normal_answer"],
    "verbosity": ["normal_answer", "short_answer"],
}

for task in task_list:
    fname_in = f"data/datasets/{task}.jsonl"
    fname_out = f"data/datasets/{task}_mod.jsonl"

    with open(fname_in, 'r') as json_file:
        json_list = list(json_file)
    with open(fname_out, "w", encoding="utf-8") as save_file:
        print("hi")
        result = []
        for i, item in enumerate(json_list):
            print(f"running on {task} {i}")
            tmp_dict =  eval(item)
            tmp_dict["positive_label"], tmp_dict["negative_label"] = pos_neg_key_label_dict[task]
            save_file.write(json.dumps(tmp_dict))
            save_file.write("\n")