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
    save_path = f"data/datasets/{task}.jsonl"
    with open(f"settings/prompts/topics/{task}.txt", 'r') as file:
        topics_line = file.read()
        topics = [topic.strip() for topic in topics_line.split(',')]
        topics_cycle = cycle(topics)

    with open(f"settings/prompts/generate_pov_themes_data/{task}.txt", "r") as file, open(save_path, "w", encoding="utf-8") as save_file:
        file_content = file.read()
        for i in tqdm(range(80)):
            topic = topics_cycle.__next__()
            vars_dict = {"topic": topic}
            prompt = file_content.format(**vars_dict)
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a JSONL dataset generator. Do not answer with anything other than JSONL."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,
                )
            content = response.choices[0].message.content
            for split in content.split("\n"):
                try:
                    tmp_dict = eval(split)
                    tmp_dict["positive_label"], tmp_dict["negative_label"] = pos_neg_key_label_dict[task]
                    save_file.write(json.dumps(tmp_dict))
                    save_file.write("\n")
                except:
                    pass

