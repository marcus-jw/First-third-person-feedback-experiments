from flask import Flask, render_template, redirect, url_for, request
import json

app = Flask(__name__)

# Path to the JSONL files
ORIGINAL_JSONL_FILE1 = 'data/hh_labels/prism_35_train_3_1.jsonl' #'datasets/new_dataset.jsonl'
ORIGINAL_JSONL_FILE3 = 'data/hh_labels/prism_35_train_3_3.jsonl' #'datasets/new_dataset.jsonl'
CLEANED_JSONL_FILE = 'data/hh_labels//cleaned_dataset.jsonl'

def read_jsonl():
    with open(ORIGINAL_JSONL_FILE1, 'r') as file1:
        lines1 = file1.readlines()
    entries1 = [json.loads(line.strip()) for line in lines1]

    with open(ORIGINAL_JSONL_FILE3, 'r') as file3:
        lines3= file3.readlines()
    entries3 = [json.loads(line.strip()) for line in lines3]

    assert len(entries1)==len(entries3), "Need the same length"
    last_question = ""
    entries = []
    for i in range(len(entries1)):
        entry = {}
        entry["conversation"] = entries1[i]["conversation"]
        entry["question"] = entries1[i]["conversation"][-1]['content']
        entry["answerA"] = entries1[i]["answer_chosen"]
        entry["answerB"] = entries1[i]["answer_rejected"]
        entry["prism_chose"] = "A"
        entry["first_chose"] = "A" if entries1[i]["logits_chosen"] > entries1[i]["logits_rejected"] else "B"
        entry["third_chose"] = "A" if entries3[i]["logits_chosen"] > entries3[i]["logits_rejected"] else "B"
        if (not (entry["prism_chose"]==entry["first_chose"]==entry["third_chose"]  )) and (entry["question"] !=last_question): # keep only the interesting entries where there is disagreement
            entries.append(entry)
        last_question = entry["question"]
    print("entries[0] ", entries[0]['question'])
    print("entries[1] ", entries[1]['question'])
    return entries

def write_jsonl(entries, filepath):
    with open(filepath, 'w') as file:
        for entry in entries:
            file.write(json.dumps(entry) + '\n')

@app.route('/')
def index():
    entries = read_jsonl()
    if 'entry_id' not in request.args or int(request.args.get('entry_id')) >= len(entries):
        return redirect(url_for('index', entry_id=0))
    entry_id = int(request.args.get('entry_id'))
    entry = entries[entry_id]
    return render_template('index.html', entry=entry, entry_id=entry_id, total_entries=len(entries))

@app.route('/next/<int:entry_id>')
def next_entry(entry_id):
    return redirect(url_for('index', entry_id=entry_id + 1))

@app.route('/delete/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    entries = read_jsonl()
    if 0 <= entry_id < len(entries):
        del entries[entry_id]
        write_jsonl(entries, CLEANED_JSONL_FILE)
    next_id = min(entry_id, len(entries) - 1)
    return redirect(url_for('index', entry_id=next_id))

if __name__ == '__main__':
    app.run(debug=True)