import torch
from safetensors.torch import load_file
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
  data = torch.load("/home/yujin-wa20/projects/aliendao/dataroot/models/mistralai/Mistral-7B-v0.1/pytorch_model-00001-of-00002.bin")
  for key in data:
    print(key, data[key].shape)
    # plot the data
    plt.figure()
    sns.heatmap(data[key].to(torch.float32).cpu().detach(), cmap='viridis')
    plt.title(f"{key}")
    plt.savefig(f'pic/{key}.png')