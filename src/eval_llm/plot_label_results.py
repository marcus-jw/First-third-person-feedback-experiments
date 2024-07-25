import json

import matplotlib.pyplot as plt
import numpy as np

# tasks = ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']
tasks = ["personalisation", "verbosity", "sycophancy", "danger_refusal", "impossible_task_refusal"]
categories = [
    "Intended personalization",
    "Tailoring of Explanations",
    "Brevity",
    "Unintended personalization",
    "Sycophancy",
    "Acceptance of Impossible Tasks",
    "Dangerous Task Complicity",
]
categories.reverse()
categories = [
    "Tailoring of Explanations",
    "Brevity",
    "Sycophancy",
    "Acceptance of Impossible Tasks",
    "Dangerous Task Complicity",
]

# categories_text_weight = ["normal","normal","normal","bold","normal", "normal", "bold"]

settings = {
    "personalisation": {"task_name": "Tailoring of Explanations", "flipped": False},
    "verbosity": {"task_name": "Brevity", "flipped": True},
    "sycophancy": {"task_name": "Sycophancy", "flipped": False},
    "impossible_task_refusal": {"task_name": "Acceptance of Impossible Tasks", "flipped": True},
    "danger_refusal": {"task_name": "Dangerous Task Complicity", "flipped": True},
}

model = ""  # gpt4o
model_postfixes = ["_gpt-4o"]  # ["_claude-3-haiku-20240307"]


data_dict = {}
for model_postfix in model_postfixes:
    for itask, task in enumerate(tasks):
        with open(f"data/datasets/{task}{model_postfix}_eval_agg.json", "r", encoding="utf-8") as f_in:
            data_dict[(task, model_postfix)] = json.load(f_in)


def flip_if_necessary(x, flipped=True):
    """
    If flipped is True compute 100 -x"""
    if flipped:
        return 100 - x
    else:
        return x


def task_name_to_yaxis_number(task_name):
    try:
        return categories.index(task_name)
    except ValueError:
        return -1


def plot_agg(tasks, model_postfix):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set the width of each bar and positions
    bar_width = 0.35
    r1 = np.arange(len(categories))
    r2 = [x + bar_width for x in r1]

    ebs = [0] * len(tasks)
    for itask, task in enumerate(tasks):

        data = data_dict[(task, model_postfix)]
        print(task, " data ", data["positive_label_prob_means"]["third_thirdshoes"])
        # cat_label = task + f'\nlow is {data["negative_label"]}, \nhigh is {data["positive_label"]}'
        task_name = settings[task]["task_name"]
        yaxis_number = task_name_to_yaxis_number(task_name)
        flipped = settings[task]["flipped"]
        print(task_name, yaxis_number)
        if itask == 1:
            # plt.errorbar(data['positive_label_prob_means']['first_firstshoes'] * 100, cat_label, c="red",
            #              xerr=data['positive_label_prob_stds']['first_firstshoes'] * 100,
            #              label="1st person 1st shoes")

            # ebs[itask] = plt.errorbar(data['positive_label_prob_means']['third_thirdshoes'] * 100, cat_label, c="blue",
            #              xerr=data['positive_label_prob_stds']['third_thirdshoes'] * 100,
            #              label="3rd person 3rd shoes")
            # ebs[itask][-1][0].set_linestyle('--')
            if True:
                ax.bar(
                    r1[yaxis_number],
                    flip_if_necessary(data["positive_label_prob_means"]["third_thirdshoes"] * 100, flipped),
                    width=bar_width,
                    color="skyblue",
                    label="3rd person Pov",
                )
                ax.bar(
                    r2[yaxis_number],
                    flip_if_necessary(data["positive_label_prob_means"]["third_firstshoes"] * 100, flipped),
                    width=bar_width,
                    color="orange",
                    label="1st person PoV",
                )
            else:
                plt.scatter(
                    data["positive_label_prob_means"]["first_firstshoes"] * 100,
                    cat_label,
                    c="red",
                    marker="o",
                    label="1st person 1st shoes",
                )
                plt.scatter(
                    data["positive_label_prob_means"]["third_thirdshoes"] * 100,
                    cat_label,
                    c="blue",
                    marker="x",
                    label="3rd person 3rd shoes",
                )
                plt.scatter(
                    data["positive_label_prob_means"]["third_firstshoes"] * 100,
                    cat_label,
                    c="blue",
                    marker="o",
                    label="3rd person 1st shoes",
                )
                plt.scatter(
                    data["positive_label_prob_means"]["first_thirdshoes"] * 100,
                    cat_label,
                    c="red",
                    marker="x",
                    label="1st person 3rd shoes",
                )
        else:
            # plt.errorbar(data['positive_label_prob_means']['first_firstshoes'] * 100, cat_label, c="red",
            #              xerr=data['positive_label_prob_stds']['first_firstshoes'] * 100)
            # ebs[itask] = plt.errorbar(data['positive_label_prob_means']['third_thirdshoes'] * 100, cat_label, c="blue", ls="--",
            #              xerr=data['positive_label_prob_stds']['third_thirdshoes'] * 100)
            # ebs[itask][-1][0].set_linestyle('--')
            if True:
                ax.bar(
                    r1[yaxis_number],
                    flip_if_necessary(data["positive_label_prob_means"]["third_thirdshoes"] * 100, flipped),
                    width=bar_width,
                    color="skyblue",
                )
                ax.bar(
                    r2[yaxis_number],
                    flip_if_necessary(data["positive_label_prob_means"]["third_firstshoes"] * 100, flipped),
                    width=bar_width,
                    color="orange",
                )
            else:
                plt.scatter(data["positive_label_prob_means"]["first_firstshoes"] * 100, cat_label, c="red", marker="o")
                plt.scatter(
                    data["positive_label_prob_means"]["third_thirdshoes"] * 100, cat_label, c="blue", marker="x"
                )
                plt.scatter(
                    data["positive_label_prob_means"]["third_firstshoes"] * 100, cat_label, c="blue", marker="o"
                )
                plt.scatter(data["positive_label_prob_means"]["first_thirdshoes"] * 100, cat_label, c="red", marker="x")

    # ax.set_title("Impact of the point of view on personalization behavior of LLM simulated feedback for " + model_postfix[1:])

    ax.set_title("PoV impact on personalization \nfor LLM simulated feedback for " + model_postfix[1:])
    ax.set_ylabel(f"Percentage of answers")
    plt.ylim(0, 100)
    ax.set_xticks([r + bar_width / 2 for r in range(len(categories))])
    ax.set_xticklabels(categories)
    ax.text(1.4, -30, "Intended Personalization", ha="right", va="center", weight="bold")
    ax.text(3.8, -30, "Unintended Personalization", ha="right", va="center", weight="bold")
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/plots/agg{model_postfix}.png")

    plt.show()


if __name__ == "__main__":
    plot_agg(tasks=tasks, model_postfix="_gpt-4o")
