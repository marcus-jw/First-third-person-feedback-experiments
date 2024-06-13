import json
opinion = False

def split_prompt(data):
    result = []
    for entry in data["prompt_list"]:
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

with open("datasets/sycophancy_fact.jsonl", 'r') as in_file, open("datasets/sycophancy_fact_fix.jsonl", 'w') as out_file:
    for line in in_file:
        data = json.loads(line)
        split_data = split_prompt(data)
        json.dump(split_data[0], out_file)
        out_file.write('\n')

