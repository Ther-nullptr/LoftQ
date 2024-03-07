import torch
from safetensors.torch import load_file
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
  data = load_file("/home/yujin-wa20/projects/LoftQ/exp_results/gsm8k_mistral_7b_4bit_64rank_loftq_None/gsm8k_mistral_7b_4bit_64rank_loftq_None/Mistral-7B-v0.1-4bit-16rank/ep_6/lr_0.0003/seed_11/adapter_model.safetensors")
  for key in data:
    print(key, data[key].shape)
    # plot the data
    plt.figure()
    sns.heatmap(data[key].cpu().detach(), cmap='viridis')
    plt.title(f"{key}_last")
    plt.savefig(f'pic/{key}_last.png')