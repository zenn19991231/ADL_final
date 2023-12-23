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
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_example', type=int, default=0)
    parser.add_argument("--which_prompt", type=str, default="promptA", required=True)
    args = parser.parse_args()
    return args

def get_prompt(which_prompt, num_example, instruction: str) -> str:

    example_path = f"/home/adl/Desktop/ADL_final/data/{which_prompt}/{which_prompt}_example.json"
    with open(example_path, "r") as file:
        example_data = json.load(file)

    random.seed(1127)
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
    # Get LoRA
    # model = PeftModel.from_pretrained(model, args.peft_path)
    return model, tokenizer

def get_data():
    # Prepare data
    with open(args.test_data_path, "r") as f:
        data = json.load(f)
    for message_item in data:
        message_item["perplexity"] = {}
        message_item["prediction"] = {}
    return data

def generate_predictions(model, tokenizer, data, max_length=2048):
    model.eval()

    # Tokenize data
    instructions = [get_prompt(args.which_prompt, args.num_example, x["instruction"]) for x in data]
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)

    # Generate prediction
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

            data[i]["prediction"].update({f"{args.num_example}_shot" :prediction})
            # if prediction == data[i]["output"]:
            #     data[i]["confident"] += 1 if prediction == data[i]["output"] else 0
    return data

def output_data(new_data):
    args.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    ppl = perplexity(model, tokenizer, data)
    ppls = ppl["perplexities"]

    if args.prediction_path.exists():
        with open(args.prediction_path, "r", encoding="utf8") as f:
            old_data = json.load(f)
        for i, item in enumerate(old_data):
            if item["id"] == new_data[i]["id"]:
                item["perplexity"].update({f"{args.num_example}_shot": ppls[i]})
        with open(args.prediction_path, "w", encoding="utf8") as f:
            json.dump(old_data, f, indent=2, ensure_ascii=False)
    else:
        for i, item in enumerate(new_data):  # Use new_data here
            if item["id"] == new_data[i]["id"]:
                item["perplexity"] = {f"{args.num_example}_shot": ppls[i]}
        with open(args.prediction_path, "w", encoding="utf8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)


    # if args.prediction_path.exists():
    #     with open(args.prediction_path, "r", encoding="utf8") as f:
    #         old_data = json.load(f)
    #     for i, item in enumerate(old_data):
    #             if item["id"] == new_data[i]["id"]:
    #                 item["prediction"].update({f"{args.num_example}_shot" :new_data[i]["prediction"][f"{args.num_example}_shot"]})
    #     with open(args.prediction_path, "w", encoding="utf8") as f:
    #         json.dump(old_data, f, indent=2, ensure_ascii=False)
    # else:
    #     with open(args.prediction_path, "w", encoding="utf8") as f:
    #         json.dump(new_data, f, indent=2, ensure_ascii=False)

def perplexity(
    model, tokenizer, data, max_length=2048,
):
    data_size = len(data)
    instructions = [get_prompt(args.which_prompt, args.num_example, x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + \
            [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + \
            output_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + \
            [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length])
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2),
             shift_label) * shift_output_mask).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

def update_ppl(model, tokenizer, data):
    ppl = perplexity(model, tokenizer, data)
    ppls = ppl["perplexities"]
    for i in range(len(data)):
        data[i]["perplexity"].update({f"{args.num_example}_shot": ppls[i]})
    # return data

if __name__ == "__main__":

    args             = parse_args()
    model, tokenizer = get_model()
    data             = get_data()
    
    model.eval()
    # ppl = perplexity(model, tokenizer, data)
    # print("Mean perplexity:", ppl["mean_perplexity"])
    # for _ in range(1):
    #     data = generate_predictions(model, tokenizer, data)
    # update_ppl(model, tokenizer, data)

    output_data(data)