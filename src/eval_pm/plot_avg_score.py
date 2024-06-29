import matplotlib.pyplot as plt
import json
import numpy as np

tasks = ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']

plt.figure(figsize=(8, 6))
perspective = "3_3"
for task in tasks:
    for perspective in ["3_3","3_1"]:
        with open(f"data/datasets/fair_{task}_{perspective}_PM_eval.jsonl", 'r',encoding="utf-8") as f_in:
            lines = f_in.readlines()
            line_0 = json.loads(lines[0])
            positive_label = line_0["positive_label"] + "_score"
            negative_label = line_0["negative_label"] + "_score"
            positive_scores = []
            negative_scores = []
            diff_scores = []
            for line in lines:
                line = json.loads(line)
                positive_scores.append(line[positive_label])
                negative_scores.append(line[negative_label])
                diff_scores.append(line[positive_label]-line[negative_label])
            med_positive_score = np.mean(positive_scores)
            med_negative_score = np.mean(negative_scores)
            med_diff_score = np.mean(diff_scores)
            if perspective == "3_3":
                plt.scatter(diff_scores, task + "\n" + positive_label.split("_")[0], c="blue", marker='o', label="3rd person 3rd shoes")
                #plt.scatter(med_negative_score, task + "\n" + positive_label.split("_")[0], c="blue", marker='x', label="3rd person 3rd shoes")
            elif perspective == "3_1":

                plt.scatter(diff_scores, task + "\n" + positive_label.split("_")[0], c="red", marker='o', label="3rd person 1st shoes")
                #plt.scatter(med_negative_score, task + "\n" + positive_label.split("_")[0], c="red", marker='x', label="3rd person 1st shoes")
            else:
                plt.scatter(med_positive_score-med_negative_score, task + "\n" + positive_label.split("_")[0], c="green", marker='o', label="none")
plt.title("Differencces in PM on synthetic datasets")
plt.xlabel(f'Average differences in PM scores')
plt.tight_layout()
plt.savefig(f'data/plots/PM_score.png')
plt.show()
