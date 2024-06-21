import matplotlib.pyplot as plt
import json

tasks = ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']
#tasks = ['sycophancy']

model = "" #gpt4o
model_postfixes = ["_gpt-3.5-turbo"]


data_dict = {}
for model_postfix in model_postfixes:
    for itask, task in enumerate(tasks):
        with open(f"data/datasets/{task}{model_postfix}_eval_agg.json", 'r', encoding="utf-8") as f_in:
            data_dict[(task, model_postfix)] = json.load(f_in)


def plot_agg(tasks, model_postfix):
    plt.figure(figsize=(8, 6))
    ebs = [0] * len(tasks)
    for itask, task in enumerate(tasks):
        data = data_dict[(task, model_postfix)]
        cat_label = task + f'\nlow is {["negative_label"]}, \nhigh is {data["positive_label"]}'

        if itask == 0:
            plt.errorbar(data['positive_label_prob_means']['first_firstshoes'] * 100, cat_label, c="red",
                         xerr=data['positive_label_prob_stds']['first_firstshoes'] * 100,
                         label="1st person 1st shoes")

            ebs[itask] = plt.errorbar(data['positive_label_prob_means']['third_thirdshoes'] * 100, cat_label, c="blue",
                         xerr=data['positive_label_prob_stds']['third_thirdshoes'] * 100,
                         label="3rd person 3rd shoes")
            ebs[itask][-1][0].set_linestyle('--')
            # plt.scatter(data['positive_label_prob_means']['first_firstshoes']*100, cat_label,c="red", marker='o', label="1st person 1st shoes")
            # plt.scatter(data['positive_label_prob_means']['third_thirdshoes']*100, cat_label, c="blue", marker='x', label="3rd person 3rd shoes")
            # plt.scatter(data['positive_label_prob_means']['third_firstshoes']*100, cat_label, c="blue", marker='o', label="3rd person 1st shoes")
            # plt.scatter(data['positive_label_prob_means']['first_thirdshoes']*100, cat_label, c="red", marker='x', label="1st person 3rd shoes")
        else:
            plt.errorbar(data['positive_label_prob_means']['first_firstshoes'] * 100, cat_label, c="red",
                         xerr=data['positive_label_prob_stds']['first_firstshoes'] * 100)
            ebs[itask] = plt.errorbar(data['positive_label_prob_means']['third_thirdshoes'] * 100, cat_label, c="blue", ls="--",
                         xerr=data['positive_label_prob_stds']['third_thirdshoes'] * 100)
            ebs[itask][-1][0].set_linestyle('--')
            # plt.scatter(data['positive_label_prob_means']['first_firstshoes'] * 100, cat_label, c="red", marker='o')
            # plt.scatter(data['positive_label_prob_means']['third_thirdshoes'] * 100, cat_label, c="blue", marker='x')
            # plt.scatter(data['positive_label_prob_means']['third_firstshoes']*100, cat_label, c="blue", marker='o')
            # plt.scatter(data['positive_label_prob_means']['first_thirdshoes']*100, cat_label, c="red", marker='x')
    plt.title("Differences in perspecitves on synthetic datasets " + model_postfix)
    plt.xlabel(f'Percentage of answers')
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/data/plots/agg_{model}.png')

if __name__ == "__main__":
    #for task in ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']:
    #for task in ['sycophancy']:
    tasks = ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']
    plot_agg(tasks=tasks, model_postfix="_gpt-3.5-turbo")