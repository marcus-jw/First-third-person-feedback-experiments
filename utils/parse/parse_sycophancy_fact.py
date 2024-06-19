import json

def parse_sycophancy_fact_entry(data, opinion):
    """
    Parse one entry for reward hacking
    :param data:
    :param opinion:
    :return: List of json with
    """
    result = []
    entry = data["prompt_list"][0] # only one entry

    entry = entry.replace('\n', ' ').strip()
    if opinion:
        main_prompt, rest = entry.split('Do you')
        rest = 'Do you' + rest
        question, answers = rest.split('Choices:')
    else:
        main_prompt, rest = entry.split('Question:')

        question, answers = rest.split('Answers:')
    answer_a, answer_b = answers.split('(B)')
    answer_a = answer_a.replace('(A)', '').strip()
    answer_b = answer_b.strip()

    result.append({"prompt": main_prompt.strip(),
        "question": question.strip(),
        "answerA": answer_a.strip(),
        "answerB": answer_b.strip(),
        "sycophantic_answer": data["high_reward_answer"],
    })
    
    return result


def parse_sycophancy_fact_dataset(fname_in, fname_out, opinion):
    """

    :param fname_in:
    :param fname_out:
    :return:
    """
    with open(fname_in, 'r') as f_in, open(fname_out, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            split_data = parse_sycophancy_fact_entry(data, opinion)
            json.dump(split_data[0], f_out)
            f_out.write('\n')

parse_sycophancy_fact_dataset("datasets/sycophancy_fact.jsonl", "datasets/sycophancy_fact_fix.jsonl", False)

