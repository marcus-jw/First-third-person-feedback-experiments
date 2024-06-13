import json
def split_prompt(data):
    result = []
    for entry in data["prompt_list"]:
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

with open("datasets/sycophancy_opinion_political.jsonl", 'r') as in_file, open("datasets/sycophancy_opinion_political_fix.jsonl", 'w') as out_file:
    for line in in_file:
        data = json.loads(line)
        split_data = split_prompt(data)
        if split_data is not None:
            json.dump(split_data[0], out_file)
            out_file.write('\n')

