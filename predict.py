import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
import argparse
from pathlib import Path
from transformers import BitsAndBytesConfig


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    # 0 shot
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

    # 3 shot
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: 翻譯成文言文：\n於是，廢帝讓瀋慶之的堂侄、直將軍瀋攸之賜瀋慶之毒藥，命瀋慶之自殺。 ASSISTANT: 帝乃使慶之從父兄子直閣將軍攸之賜慶之藥。 USER: 文言文翻譯：\n靈鑒忽臨，忻歡交集，乃迴燈拂席以延之。 ASSISTANT: 答案：靈仙忽然光臨，趙旭歡欣交集，於是他就把燈點亮，拂拭乾淨床席來延請仙女。 USER: 希望您以後留意，不要再齣這樣的事，你的小女兒病就會好。\n這句話在古代怎麼說： ASSISTANT: 以後幸長官留意，勿令如此。 USER: {instruction} ASSISTANT:"




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
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
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
    args = parser.parse_args()
    return args

def keep_after_last_colon(text):
    last_colon_index = text.rfind(':')
    return text[last_colon_index + 1:].strip()

def get_ready():
    # Get model
    bnb_config = get_bnb_config()
    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    # Get tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)

    # Get data
    with open(args.test_data_path, "r") as f:
        data = json.load(f)
    
    return model, tokenizer, data

def generate_predictions(model, tokenizer, data, max_length=2048):
    model.eval()

    # Tokenize data
    instructions = [get_prompt(x["instruction"]) for x in data]
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)

    prediction_output = []
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
            prediction_output.append({
                "id":data[i]["id"],
                "output":prediction
            })

    args.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.prediction_path, "w", encoding="utf8") as f:
        json.dump(prediction_output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    
    model, tokenizer, data = get_ready()
    
    generate_predictions(model, tokenizer, data)