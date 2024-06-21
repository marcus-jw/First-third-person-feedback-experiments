from openai import OpenAI
import os
import math
import matplotlib.pyplot as plt
import json
import random
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import concurrent.futures


def get_cl_margin_of_error(samples, confidence_level = 0.95):
    """
    Get confidence level margin or error
    :return: margin of error
    """

    # Calculate sample mean
    mean = np.mean(samples)

    # Calculate sample standard deviation
    std_dev = np.std(samples, ddof=1)

    # Calculate standard error of the mean
    sem = std_dev / np.sqrt(len(samples))

    # Determine the t-critical value for 95% confidence interval
    degrees_freedom = len(samples) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

    # Calculate margin of error
    margin_of_error = t_critical * sem

    # Calculate confidence interval
    #cl_lower, cl_upper = (mean - margin_of_error, mean + margin_of_error)

    #print(f"The 95% confidence interval is {confidence_interval}")
    return margin_of_error

def aggregate_llm(model="gpt-4o", task="sycophancy"):
    # Load the JSON file
    model_postfix = "" if model == "" else "_" + model
    fname_in = f'data/datasets/{task}{model_postfix}_eval.jsonl'
    fname_out = f'data/datasets/{task}{model_postfix}_eval_agg.json'
    with open(fname_in, 'r', encoding="utf-8") as f_in, open(fname_out, 'w', encoding="utf-8") as f_out:
        data = []
        for line in f_in:
            data.append(json.loads(line))

        keys = data[0]["positive_label_prob"].keys()


        results = {
            "task": task,
            "settings": {
                #"scratchpad": SCRATCHPAD,
                "model": model,
            },
            "positive_label": data[0]["positive_label"],
            "negative_label": data[0]["negative_label"],
            "positive_label_prob_means": {key: np.mean([row["positive_label_prob"][key] for row in data]) for key in keys},
            "positive_label_prob_stds": {key: get_cl_margin_of_error([row["positive_label_prob"][key] for row in data]) for key in keys},
            "negative_label_prob_means": {key: np.mean([row["negative_label_prob"][key] for row in data]) for key in keys},
            "negative_label_prob_stds": {key: get_cl_margin_of_error([row["negative_label_prob"][key] for row in data]) for key in keys},
        }

        json.dump(results, f_out)



if __name__ == "__main__":
    for task in ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']:
    #for task in ['sycophancy']:
        aggregate_llm(task=task, model="gpt-3.5-turbo")