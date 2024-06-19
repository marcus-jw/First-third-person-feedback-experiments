from openai import OpenAI
import json
from tqdm import tqdm
from itertools import cycle
client = OpenAI()

model = "gpt-4o"
#task = "sycophancy"
#task = "impossible_task"
task = "danger_refusal"
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
                save_file.write(json.dumps(eval(split)))
                save_file.write("\n")
            except:
                pass
        
        