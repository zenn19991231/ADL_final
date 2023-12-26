# ADL_final

## 詐騙簡訊偵測Demo
```
python ./src/Demo.py
```

## Project Structure
* /data: 測試與驗證之簡訊資料集
* /accuracy: 計算label之正向度rating之相關檔案
* /perplexity: 計算perplexity之相關檔案
* /rouge_score: 計算rouge_score之相關檔案
* /src: Source code
  * Demo.py: 詐騙簡訊偵測Demo程式
  * predict.py: 產生不同prompt在不同shot之下的結果
 
## 執行predict.py
chose PromptA, PromptB, PromptC for {which_propt}
```
python3           "predict.py" \
--base_model_path  "/home/adl/Desktop/ADL_final/model/Taiwan-LLM-7B-v2.0-chat" \
--test_data_path   "data/{which_prompt}/{which_prompt}_combine.json" \
--prediction_path  "data/{which_prompt}/{which_prompt}_predict.json" \
--which_prompt     "{which_prompt}" \
--device_id        1
```
