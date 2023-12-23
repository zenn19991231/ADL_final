import torch
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from peft import PeftModel
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_prompt(which_prompt, num_example, instruction: str) -> str:

    example_path = f"data/{which_prompt}/{which_prompt}_example.json"
    with open(example_path, "r") as file:
        example_data = json.load(file)

    random.seed(2096)
    example_data = random.sample(example_data, len(example_data))

    promptA = "你是一個精明的偵探，現在你要幫忙審視以下訊息的真偽，你會提供重要的關鍵資訊作為依據，給每一行一則訊息作出評價，評價方式為：\“訊息真偽；可信度0~100(越高越可信)；原因。\“"
    promptB = "以準確的關鍵資訊作為判斷依據一句話簡短描述以下每則訊息內容的真偽，並依據其真偽程度評分0到100，並依照此格式回答：真偽程度。原因。"
    promptC = "你是詐騙偵測機器人，請評論以下訊息"
    prefixA = "請幫忙審視以下訊息的真偽"
    prefixB = "以準確的關鍵資訊作為判斷依據一句話簡短描述以下每則訊息內容的真偽"
    prefixC = "你是詐騙偵測機器人，請評論以下訊息"

    if which_prompt == "promptA":
        prompt = promptA
        prefix = prefixA
    elif which_prompt == "promptB":
        prompt = promptB
        prefix = prefixB
    elif which_prompt == "promptC":
        prompt = promptC
        prefix = prefixC

    examples = [prompt]
    for i in range(num_example):
        user      = example_data[i]["instruction"]
        assistant = example_data[i]["output"]
        user      = f"USER: {prefix}\n{user}"
        assistant = f"ASSISTANT: {assistant}"
        examples.append(user)
        examples.append(assistant)

    example = f"USER: {prefix}\n{instruction} ASSISTANT: "
    examples.append(example)
    return " ".join(examples)

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--prediction_path",
        type=Path,
        default="",
        required=True,
        help="Path to prediction."
    )
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--which_prompt", type=str, default="promptA", required=True)
    args = parser.parse_args()
    return args

def keep_after_last_colon(text):
    last_colon_index = text.rfind(':')
    return text[last_colon_index + 1:].strip()

def get_model():

    torch.cuda.set_device(args.device_id)

    # Get model & tokenizer
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def get_data(shots):
    # Prepare data
    with open(args.test_data_path, "r") as f:
        data = json.load(f)
    for message_item in data:
        message_item['prediction'] = {}
    return data

def generate_predictions(shots, model, tokenizer, data, max_length=2048):
    model.eval()

    for num_example in shots:
        key = f"{num_example}_shot"

        # Tokenize data
        instructions = [get_prompt(args.which_prompt, num_example, x["instruction"]) for x in data]
        tokenized_instructions = tokenizer(instructions, add_special_tokens=False)

        # Generate prediction
        for i in tqdm(range(len(data))):
            input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
            input_ids = torch.tensor(input_ids[:max_length]).unsqueeze(0)

            with torch.no_grad():
                prediction = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=0.75
                )
                prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
                prediction = keep_after_last_colon(prediction[0])
                data[i]["prediction"][key] = prediction
    return data

def output_data(data):
    args.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.prediction_path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":

    shots = [0, 1, 2, 3, 4, 8, 12, 16, 20]

    args             = parse_args()
    model, tokenizer = get_model()
    data             = get_data(shots)

    data = generate_predictions(shots, model, tokenizer, data)

    output_data(data)