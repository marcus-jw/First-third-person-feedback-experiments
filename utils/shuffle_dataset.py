import json
import random

def randomize_order_and_choice_for_datasets(fname_in, fname_out):
    """
    When GPT returns a generated dataset there could be a difference between what was generated as answer A vs answer B (choice).
    This code shuffles this choice and randomizes the order
    :return:
    """
    with open(fname_in, 'r', encoding="utf-8") as f_in, open(fname-out, 'w', encoding="utf-8") as f_out:
        # load data; can be more efficient
        data = []
        for line in in_file:
            data.append(json.loads(line))

        # Shuffle the data
        random.shuffle(data)

        for i in range(0, len(data)):
            l = random.random()
            if l < 0.5:
                data[i]["answerA"],data[i]["answerB"] = data[i]["answerB"],data[i]["answerA"]
                data[i]["sycophantic_answer"] = "A" if data[i]["sycophantic_answer"] == "B" else "B"

            out = json.dumps(data[i])
            out_file.write(out + '\n')

    