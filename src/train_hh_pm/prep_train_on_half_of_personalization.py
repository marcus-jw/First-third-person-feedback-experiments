import json

import numpy as np
strategy = "_fortest" # "_fortraining" # "_fortest" # ""
ONLY_INTENDED = False

if ONLY_INTENDED:
    tasks = ["personalisation", "verbosity"]
    ds_prefix = "intended"
else:
    tasks = ["personalisation", "verbosity", "sycophancy", "danger_refusal", "impossible_task_refusal"]
    ds_prefix = "all"
model_postfix = "_gpt-4o" #"_gpt-3.5-turbo"

for perspective in ["3_1", "3_3"]:
    perspective_name = "third_firstshoes" if perspective == "3_1" else "third_thirdshoes"
    data_all = []
    for itask, task in enumerate(tasks):
        fname_in = f"data/datasets/{task}{model_postfix}_eval.jsonl"
        with open(fname_in, "r", encoding="utf-8") as f_in:
            data = []
            for line in f_in:
                # print("line ", line)
                d = json.loads(line)
                d["answer_chosen"] = d[d["positive_label"]]
                d["answer_rejected"] = d[d["negative_label"]]
                d["logits_chosen"] = np.log(d["positive_label_prob"][perspective_name])
                d["logits_rejected"] = np.log(d["negative_label_prob"][perspective_name])
                d = {
                    key: d[key]
                    for key in [
                        "prompt",
                        "question",
                        "answer_chosen",
                        "answer_rejected",
                        "logits_chosen",
                        "logits_rejected",
                    ]
                }
                data.append(d)
            if strategy=="_fortraining":
                data = data[: len(data) // 2]
            elif strategy=="_fortest":
                data = data[len(data) // 2 : ((len(data) // 2) + (len(data) // 50))]
            data_all.extend(data)

    with open(f"data/datasets/{ds_prefix}_personalization_{perspective}{model_postfix}{strategy}.jsonl", "w") as outfile:
        for entry in data_all:
            json.dump(entry, outfile)
            outfile.write("\n")
