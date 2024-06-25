from openai import OpenAI
import os
import math
import matplotlib.pyplot as plt
import json
import random
import numpy as np
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

def aggregate_llm(task="sycophancy"):
    # Load the JSON file
    fname_in = f'data/datasets/{task}_3_3_PM_eval.jsonl'
    fname_out = f'data/datasets/{task}_3_3_PM_eval_agg.json'
    with open(fname_in, 'r', encoding="utf-8") as f_in, open(fname_out, 'w', encoding="utf-8") as f_out:
        data = []
        for line in f_in:
            data.append(json.loads(line))

        positive_label_score = data[0]["positive_label"] + "_score"
        negative_label_score = data[0]["negative_label"] + "_score"

        results = {
            "task": task,
            "positive_label": data[0]["positive_label"],
            "positive_label_prob_means": np.mean([row[positive_label_score]/(row[positive_label_score]+row[negative_label_score]) for row in data]),
        }

        print(results)



if __name__ == "__main__":
    for task in ['danger_refusal', 'impossible_task_refusal', 'personalisation', 'sycophancy', 'verbosity']:
    #for task in ['sycophancy']:
        aggregate_llm(task=task)