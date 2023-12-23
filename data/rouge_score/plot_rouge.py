import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--rouge')
parser.add_argument('-o', '--rouge_fig')
args = parser.parse_args()

with open(args.rouge, 'r') as file:
    data = json.load(file)

# 不同 shot 數量的值（rouge-1、rouge-2、rouge-l的r、p、f）
shots = ["0_shot", "1_shot", "2_shot", "3_shot", "4_shot"]  # 可根據資料量增減

# 三種rouge的指標
rouge_metrics = ["rouge-1", "rouge-2", "rouge-l"]

# 將r、p、f值分離到不同的列表中
r_values = {metric: [data[shot][metric]["r"] for shot in shots] for metric in rouge_metrics}
p_values = {metric: [data[shot][metric]["p"] for shot in shots] for metric in rouge_metrics}
f_values = {metric: [data[shot][metric]["f"] for shot in shots] for metric in rouge_metrics}

bar_width = 0.2
index = np.arange(len(shots))

# 繪製長條圖
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

for i, metric in enumerate(rouge_metrics):
    r_bars = axs[i].bar(index - bar_width, r_values[metric], bar_width, alpha=0.7, label="R",color="black")
    p_bars = axs[i].bar(index, p_values[metric], bar_width, alpha=0.7, label="P")
    f_bars = axs[i].bar(index + bar_width, f_values[metric], bar_width, alpha=0.7, label="F")
    axs[i].set_ylabel(f"{metric.upper()} Values")
    axs[i].set_xlabel("Shots")
    axs[i].set_title(f"{metric.upper()} Comparison Across Shots")
    axs[i].set_xticks(index)
    axs[i].set_xticklabels(shots)
    axs[i].legend()
    
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # 調整垂直方向的間距
# 儲存圖表為圖像檔案
plt.savefig(args.rouge_fig, bbox_inches='tight')  # bbox_inches='tight' 可以裁剪圖表周圍的空白部分




