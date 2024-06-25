import matplotlib.pyplot as plt
import json

tasks = ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']

plt.figure(figsize=(8, 6))
perspective = "3_3"
for task in tasks:
    for perspective in ["3_3","3_1","none"]:
        with open(f"data/datasets/{task}_{perspective}_PM_eval.jsonl", 'r',encoding="utf-8") as f_in:
            lines = f_in.readlines()
            line_0 = json.loads(lines[0])
            positive_label = line_0["positive_label"] + "_score"
            negative_label = line_0["negative_label"] + "_score"
            p_count = 0
            for line in lines:
                line = json.loads(line)
                if line[positive_label]>line[negative_label]:
                    p_count += 1
            prob = p_count/len(lines)
            if perspective == "3_3":
                plt.scatter(prob*100, task + "\n" + positive_label.split("_")[0], c="blue", marker='o', label="3rd person 3rd shoes")
            elif perspective == "3_1":

                plt.scatter(prob*100, task + "\n" + positive_label.split("_")[0], c="red", marker='o', label="3rd person 1st shoes")
            else:
                plt.scatter(prob*100, task + "\n" + positive_label.split("_")[0], c="green", marker='o', label="none")
plt.title("Differencces in PM on synthetic datasets")
plt.xlabel(f'Percentage of answers')
plt.xlim(0,100)
plt.legend()
plt.tight_layout()
plt.savefig(f'data/plots/PM_eval.png')
plt.show()