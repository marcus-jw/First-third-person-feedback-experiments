import json
def parse_sycophancy_opinion_political_entry(data, opinion):
    result = []
    entry = data["prompt_list"][0]
    if "?" in entry:
        start, end = entry.split("?")
        start_parts = start.split(".")
        main_prompt = ".".join(start_parts[:-1]) + "."
        question = start_parts[-1] + "?"
        answer_a, answer_b = end.split('(B)')
        answer_a = answer_a.replace('(A)', '').strip()
        answer_b = answer_b.strip()

        result.append({"prompt": main_prompt.strip(),
            "question": question.strip(),
            "answerA": answer_a.strip(),
            "answerB": answer_b.strip(),
            "sycophantic_answer": data["high_reward_answer"],
        })
    else:
        return None
    
    return result

def parse_sycophancy_opinion_political_dataset(fname_in, fname_out, opinion):
    """

    :param fname_in:
    :param fname_out:
    :return:
    """
    for line in in_file:
        data = json.loads(line)
        split_data = split_prompt(data)
        if split_data is not None:
            json.dump(split_data[0], out_file)
            out_file.write('\n')

parse_sycophancy_fact_dataset("datasets/sycophancy_opinion_political.jsonl", "datasets/sycophancy_opinion_political_fix.jsonl", False)