import json

# 读取 JSON 文件
with open('data/promptC/promptA_predict.json', 'r') as json_file:
    data = json.load(json_file)

# 写入 JSONL 文件
with open('data/promptC/promptA_predict.jsonl', 'w') as jsonl_file:
    for item in data:
        json.dump(item, jsonl_file)
        jsonl_file.write('\n')
