import json
import numpy as np
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
# Load predictions from a file
train_test = "train"
split = "harmless-base"
mapping = {"harmless-base": (0,42537),
            "helpful-base": (42538, 86372),
            "helpful-online": (86373, 108379),
            "helpful-rejection-sampled": (108380, 160800)}

# "harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"
def load_perspective_predictions(in_filepath):
    with open(in_filepath, 'r', encoding='utf-8') as in_file:
        start = mapping[split][0]
        stop = mapping[split][1]
        part = [line for i, line in enumerate(in_file) if start <= i <= stop]
        preds = [float(json.loads(line)["prob_chosen"]) > float(json.loads(line)["prob_rejected"]) for line in part]
        #preds = [randint(0,1) for line in in_file]
    return np.array([1 if pred else 0 for pred in preds])

def calculate_agreement_proportions(filepaths):
    # Load predictions for all files
    predictions = {filepath: load_perspective_predictions(filepath) for filepath in filepaths}
    print([len(pred) for pred in predictions.values()])
    filenames = list(predictions.keys())
    agreement_matrix = np.zeros((len(filenames)+1, len(filenames)+1))
    preds_anthropic = np.ones(len(next(iter(predictions.values()))))
    for i, (file1, preds1) in enumerate(predictions.items()):
        for j, (file2, preds2) in enumerate(predictions.items()):

            agreement = np.mean(preds1 == preds2)
            agreement_matrix[i, j] = agreement
    for i in range(len(filenames)):
        agreement_matrix[i, -1] = np.mean(preds_anthropic == predictions[filenames[i]])
        agreement_matrix[-1, i] = agreement_matrix[i, -1]
        agreement_matrix[-1, -1] = 1
    # Mapping filepaths to a simpler name
    simplified_names = ["1 imagining 1", "1 imagining 3", "3 imagining 1", "3 imagining 3"]
    simplified_names.append(split)
    return agreement_matrix, simplified_names

base = f"data/hh_labels/haiku_{split}_{train_test}_"
filepaths = [
    base + "1_1.jsonl",
    base + "1_3.jsonl",
    base + "3_1.jsonl",
    base + "3_3.jsonl",
    #"data/datasets/hh-rlhf-train-extracted.jsonl",
]


agreement_matrix, simplified_names = calculate_agreement_proportions(filepaths)

plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=simplified_names, yticklabels=simplified_names)
plt.title("Correlations between perspective feedback")
plt.tight_layout()
plt.show()