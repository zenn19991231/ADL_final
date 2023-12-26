import torch
from transformers import pipeline
from huggingface_hub import login
import random

login(token="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")

pipe = pipeline("text-generation", model="yentinglin/Taiwan-LLM-13B-v2.0-chat", torch_dtype=torch.bfloat16, device_map="auto")
classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-full-chinese")

prompt_list = {
    "promptA" : "你是一個精明的偵探，現在你要幫忙審視以下訊息的真偽，你會提供重要的關鍵資訊作為依據，給每一行一則訊息作出評價，評價方式為：“訊息真偽；可信度0~100(越高越可信)；原因。",
    "promptB" : "以準確的關鍵資訊作為判斷依據一句話簡短描述以下每則訊息內容的真偽，並依據其真偽程度評分0到100，並依照此格式回答：真偽程度。原因。",
    "promptC" : "你是詐騙偵測機器人，請評論以下訊息。USER: 嗨！好久不見，最近過得如何？有空一起出來喝杯咖啡嗎？ \
ASSIS: 這是一個一般性的社交邀請，看起來沒有任何詐騙的跡象。 \
USER: 您的帳戶已被鎖定，必須立即點擊以下連結重設密碼，否則將會永久關閉。\
ASSIS: 這可能是釣魚郵件，企圖取得您的帳戶資訊。真正的服務商通常會提供其他方法來解鎖帳戶，而不會要求直接點擊連結。"
}
selected_prompt = prompt_list["promptB"]

def randomize_dict_order(input_dict):
    # 取得字典的鍵(keys)
    keys = list(input_dict.keys())

    # 隨機排列字典的鍵
    random.shuffle(keys)

    # 創建一個新的字典，根據隨機排序的鍵重新排序原始字典
    randomized_dict = {key: input_dict[key] for key in keys}
    return randomized_dict

def find_median_name(scores_dict):
    # 從字典中提取分數
    scores = list(scores_dict.values())
    scores.sort()  # 排序分數

    n = len(scores)
    if n % 2 == 0:
        # 若分數個數為偶數
        mid_right = n // 2
        mid_left = mid_right - 1
        median_score = (scores[mid_left] + scores[mid_right]) / 2
    else:
        # 若分數個數為奇數
        mid = n // 2
        median_score = scores[mid]

    # 找到中位數對應的名字
    median_name = [name for name, score in scores_dict.items() if score == median_score]

    return median_name,median_score


while(True):
    text = input("\n請輸入欲偵測之簡訊，或輸入Ｑ退出：\n")
    results = []
    if text and text!= "Q" and text!= "q":
        print("\n偵測中，請稍候...\n")
        messages1 = [
            {
                "role": "system",
                "content":prompt_list["promptA"] ,
            },
            {"role": "user", "content": text},
        ]
        messages2 = [
            {
                "role": "system",
                "content": prompt_list["promptB"],
            },
            {"role": "user", "content": text},
        ]
        messages3 = [
            {
                "role": "system",
                "content": prompt_list["promptC"],
            },
            {"role": "user", "content": text},
        ]
        keyword = "ASSISTANT:"   
        prompt1 = pipe.tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        outputs1 = pipe(prompt1, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        index1 = outputs1[0]['generated_text'].find(keyword)
        prompt2 = pipe.tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        outputs2 = pipe(prompt2, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        index2 = outputs2[0]['generated_text'].find(keyword)
        prompt3 = pipe.tokenizer.apply_chat_template(messages3, tokenize=False, add_generation_prompt=True)
        outputs3 = pipe(prompt3, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        index3 = outputs3[0]['generated_text'].find(keyword)
        if index1 != -1:
            # 找到詞句後，取得該詞句之後的字串
            result1 = outputs1[0]['generated_text'][index1 + len(keyword):].strip()
        if index2 != -1:
            # 找到詞句後，取得該詞句之後的字串
            result2 = outputs2[0]['generated_text'][index2 + len(keyword):].strip()
        if index3 != -1:
            # 找到詞句後，取得該詞句之後的字串
            result3 = outputs3[0]['generated_text'][index3 + len(keyword):].strip()
        results = [result1,result2,result3]
        ratings = [ int(item['label'].split(' ')[1]) for item in classifier(results) ]
        result_dict = dict(zip(results, ratings))
        print(result_dict)
        randomized_result_dict = randomize_dict_order(result_dict)
        final_result,final_score = find_median_name(randomized_result_dict)
        star = ""
        for i in range(final_score):
            star = star + "⭐️"
        print("------------------------------------------------------------------------------")
        print("\n偵測結果：\n")
        print("正向度Rating:",star)
        print(final_result[0])
        print("------------------------------------------------------------------------------")
        
        
    if text == "Q" or text == "q":
        break
    else:
        continue

