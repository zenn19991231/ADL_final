import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--json')
parser.add_argument('-o', '--jsonl')
args = parser.parse_args()

with open(args.json, 'r') as json_file:
    data = json.load(json_file)

with open(args.jsonl, 'w') as jsonl_file:
    for item in data:
        json.dump(item, jsonl_file)
        jsonl_file.write('\n')

