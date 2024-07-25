import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# tasks = ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']
tasks = ["personalisation", "verbosity", "sycophancy", "danger_refusal", "impossible_task_refusal"]
categories = [
    "Tailoring of Explanations",
    "Brevity",
    "Sycophancy",
    "Acceptance of Impossible Tasks",
    "Dangerous Task Complicity",
]
PLOT_THREE_BARS = False
perspectives = ["3_3", "3_1", "untrained"] if PLOT_THREE_BARS else ["3_3", "3_1"]  # perspect none is unfinetuned
pm_iterations = 3
# model_specifier = "_trainonperso_grm_big_10epochs"
model_specifier = "_trainonperso_grm_big_1epochs_lr5e-4"


def Wilson_mean_stdev(series_a, series_b):
    comparison = series_a > series_b
    successes = np.sum(comparison)
    n = len(series_a)
    fraction = successes / n
    # Compute the Wilson score interval (95% confidence level)
    z = stats.norm.ppf(0.975)  # 0.975 for 95% CI
    denominator = 1 + z**2 / n
    p_tilde = (fraction + z**2 / (2 * n)) / denominator
    standard_error = np.sqrt((fraction * (1 - fraction) + z**2 / (4 * n)) / n) / denominator
    mean = p_tilde
    stdev = z * standard_error
    return mean, stdev


data_dict = {}
for perspective in perspectives:
    for itask, task in enumerate(tasks):
        probs = []
        for pm_iteration in range(1, 1 + pm_iterations):
            with open(
                f"data/datasets/{task}_{perspective}_PM_eval{model_specifier}{pm_iteration}.jsonl",
                "r",
                encoding="utf-8",
            ) as f_in:
                lines = f_in.readlines()
                line_0 = json.loads(lines[0])
                positive_label = line_0["positive_label"] + "_score"
                negative_label = line_0["negative_label"] + "_score"
                p_count = 0
                series_positive, series_negative = [], []
                for line in lines:
                    line = json.loads(line)
                    series_positive.append(line[positive_label])
                    series_negative.append(line[negative_label])

                series_positive = np.array(series_positive)
                series_negative = np.array(series_negative)

                prob = np.mean(series_positive > series_negative)
                # prob_mean, stdev= Wilson_mean_stdev(series_positive, series_negative)
                probs.append(prob)
        # print(f"task {task} probs {probs}")
        probs = np.array(probs)
        print(f"{task} {perspective}", probs)
        data_dict[(task, perspective)] = (probs.mean() * 100, probs.mean() * 100, probs.std() * 100)


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


settings = {
    "personalisation": {"task_name": "Tailoring of Explanations", "flipped": False},
    "verbosity": {"task_name": "Brevity", "flipped": True},
    "sycophancy": {"task_name": "Sycophancy", "flipped": False},
    "impossible_task_refusal": {"task_name": "Acceptance of Impossible Tasks", "flipped": True},
    "danger_refusal": {"task_name": "Dangerous Task Complicity", "flipped": True},
}

perspective_to_label = {
    "3_3": "3rd person Pov",
    "3_1": "3rd person Pov",
}


fig, ax = plt.subplots(figsize=(8, 6))
# Set the width of each bar and positions
bar_width = 0.25 if PLOT_THREE_BARS else 0.3
r1 = np.arange(len(categories))
r2 = [x + bar_width for x in r1]
if PLOT_THREE_BARS:
    r3 = [x + 2 * bar_width for x in r1]

for itask, task in enumerate(tasks):

    task_name = settings[task]["task_name"]
    yaxis_number = task_name_to_yaxis_number(task_name)
    flipped = settings[task]["flipped"]
    if itask == 1:
        ax.bar(
            r1[yaxis_number],
            flip_if_necessary(data_dict[(task, "3_3")][0], flipped),
            width=bar_width,
            color="skyblue",
            label="3rd person Pov",
        )
        ax.errorbar(
            r1[yaxis_number],
            flip_if_necessary(data_dict[(task, "3_3")][1], flipped),
            yerr=data_dict[(task, "3_3")][2],
            color="k",
        )
        ax.bar(
            r2[yaxis_number],
            flip_if_necessary(data_dict[(task, "3_1")][0], flipped),
            width=bar_width,
            color="orange",
            label="1st person PoV",
        )
        ax.errorbar(
            r2[yaxis_number],
            flip_if_necessary(data_dict[(task, "3_1")][1], flipped),
            yerr=data_dict[(task, "3_1")][2],
            color="k",
        )
        if PLOT_THREE_BARS:
            ax.bar(
                r3[yaxis_number],
                flip_if_necessary(data_dict[(task, "untrained")][0], flipped),
                width=bar_width,
                color="grey",
                label="Untrainedd",
            )

    else:
        ax.bar(
            r1[yaxis_number], flip_if_necessary(data_dict[(task, "3_3")][0], flipped), width=bar_width, color="skyblue"
        )
        ax.errorbar(
            r1[yaxis_number],
            flip_if_necessary(data_dict[(task, "3_3")][1], flipped),
            yerr=data_dict[(task, "3_3")][2],
            color="k",
        )
        ax.bar(
            r2[yaxis_number], flip_if_necessary(data_dict[(task, "3_1")][0], flipped), width=bar_width, color="orange"
        )
        ax.errorbar(
            r2[yaxis_number],
            flip_if_necessary(data_dict[(task, "3_1")][1], flipped),
            yerr=data_dict[(task, "3_1")][2],
            color="k",
        )
        if PLOT_THREE_BARS:
            ax.bar(
                r3[yaxis_number],
                flip_if_necessary(data_dict[(task, "untrained")][0], flipped),
                width=bar_width,
                color="grey",
            )

    ax.set_title(f"PoV impact on personalization \nfor PM trained on HH with GPT-4o labels\n{model_specifier}")
    ax.set_ylabel(f"Percentage of answers")
    plt.ylim(0, 100)
    plt.xlim(-1 * bar_width, len(categories) - 0.5 * bar_width)
    ax.set_xticks([r + bar_width / 2 * 2 for r in range(len(categories))])
    ax.set_xticklabels(categories)
    ax.text(1.4, -30, "Intended Personalization", ha="right", va="center", weight="bold")
    ax.text(3.8, -30, "Unintended Personalization", ha="right", va="center", weight="bold")
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    fname_fig = f"data/plots/pm{model_specifier}.png"
    print(f"Saving to {fname_fig}")
    plt.savefig(fname_fig)

    plt.show()


if False:
    plt.figure(figsize=(8, 6))
    perspective = "3_3"
    for task in tasks:
        for perspective in ["3_3", "3_1", "none"]:
            with open(f"data/datasets/{task}_{perspective}_PM_eval.jsonl", "r", encoding="utf-8") as f_in:
                lines = f_in.readlines()
                line_0 = json.loads(lines[0])
                positive_label = line_0["positive_label"] + "_score"
                negative_label = line_0["negative_label"] + "_score"
                p_count = 0
                for line in lines:
                    line = json.loads(line)
                    if line[positive_label] > line[negative_label]:
                        p_count += 1
                prob = p_count / len(lines)
                if perspective == "3_3":
                    plt.scatter(
                        prob * 100,
                        task + "\n" + positive_label.split("_")[0],
                        c="blue",
                        marker="o",
                        label="3rd person 3rd shoes",
                    )
                elif perspective == "3_1":

                    plt.scatter(
                        prob * 100,
                        task + "\n" + positive_label.split("_")[0],
                        c="red",
                        marker="o",
                        label="3rd person 1st shoes",
                    )
                else:
                    plt.scatter(
                        prob * 100, task + "\n" + positive_label.split("_")[0], c="green", marker="o", label="none"
                    )
    plt.title("Differencces in PM on synthetic datasets")
    plt.xlabel(f"Percentage of answers")
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/plots/PM_eval.png")
    plt.show()
