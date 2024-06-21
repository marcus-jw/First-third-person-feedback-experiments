import matplotlib.pyplot as plt
import json

tasks = ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']

plt.figure(figsize=(8, 6))
for task in tasks:
    with open(f"data/datasets/{task}_eval.json", 'r',encoding="utf-8") as f_in:
        data = json.load(f_in)
        #print(data)
        cat_label = task + f'\nlow is {data["negative_label"]}, \nhigh is {data["positive_label"]}'
        plt.scatter(data['probs_positive']['first_firstshoes']*100, cat_label,c="red", marker='o', label="1st person 1st shoes")
        plt.scatter(data['probs_positive']['third_thirdshoes']*100, cat_label, c="blue", marker='x', label="3rd person 3rd shoes")
        plt.scatter(data['probs_positive']['third_firstshoes']*100, cat_label, c="blue", marker='o', label="3rd person 1st shoes")
        plt.scatter(data['probs_positive']['first_thirdshoes']*100, cat_label, c="red", marker='x', label="1st person 3rd shoes")
plt.title("Differencces in perspecitves on synthetic datasets")
plt.xlabel(f'Percentage of answers')
plt.xlim(0,100)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f'/data/plots/{task}.png')
