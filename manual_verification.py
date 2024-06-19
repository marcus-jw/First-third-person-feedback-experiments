from flask import Flask, render_template, redirect, url_for, request
import json

app = Flask(__name__)

# Path to the JSONL files
ORIGINAL_JSONL_FILE = 'datasets/new_dataset.jsonl'
CLEANED_JSONL_FILE = 'datasets/cleaned_dataset.jsonl'

def read_jsonl():
    with open(ORIGINAL_JSONL_FILE, 'r') as file:
        lines = file.readlines()
    entries = [json.loads(line.strip()) for line in lines]
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