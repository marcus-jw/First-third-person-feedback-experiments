import json

def merge_jsonl_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for file in input_files:
            with open(file, 'r') as infile:
                for line in infile:
                    # Check if the line is a list
                    if line.strip().startswith('['):
                        json_list = json.loads(line.strip())
                        for json_obj in json_list:
                            outfile.write(json.dumps(json_obj) + '\n')
                    else:
                        json_line = json.loads(line.strip())
                        outfile.write(json.dumps(json_line) + '\n')

if __name__ == "__main__":
    # List of types and names
    types = ["1_1", "3_1", "1_3", "3_3"]
    names = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
    
    for type_ in types:
        input_files = [f'data/hh_labels/{name}_train_{type_}.jsonl' for name in names]
        output_file = f'data/hh_labels/hh_train_{type_}.jsonl'
        merge_jsonl_files(input_files, output_file)
        print(f"Merged {len(input_files)} files into {output_file}")
