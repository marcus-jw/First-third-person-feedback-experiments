import json
import random
with open("datasets/opinion.jsonl", 'r', encoding="utf-8") as in_file, open("datasets/new_opinion.jsonl", 'w', encoding="utf-8") as out_file:
    data=[]
    for line in in_file:
        data.append(json.loads(line))
    # Shuffle the data
    random.shuffle(data)
    for i in range(0, len(data)):
        l = random.random()
        if l < 0.5:
            data[i]["answerA"],data[i]["answerB"] = data[i]["answerB"],data[i]["answerA"]
            data[i]["sycophantic_answer"] = "A" if data[i]["sycophantic_answer"] == "B" else "B"

        out = json.dumps(data[i])
        out_file.write(out + '\n')

    