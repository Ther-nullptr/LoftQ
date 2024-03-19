import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    datadir = '/home/yujin-wa20/projects/LoftQ/output'
    files = os.listdir(datadir)
    files = [file for file in files if 'o_proj.lora_A' in file and '.png' not in file]
    # sort by the number
    files = sorted(files, key = lambda s: int(s.split('.')[4]))
    
    for i in range(0, 31):
        print(f'data {i}')
        print(files[i])
        data1 = torch.load(f'{datadir}/{files[i]}').cpu().detach().numpy() # 
        data2 = torch.load(f'{datadir}/{files[i+ 1]}').cpu().detach().numpy()
        diff = data2 - data1

        # # print the norm of data1 & diff
        # norm_data1 = torch.sum(torch.abs(data1))
        # norm_data2 = torch.sum(torch.abs(diff))
        # print(f'norm of data1: {norm_data1}')
        # print(f'norm of diff: {norm_data2}')
        # print(f'ratio: {norm_data2 / norm_data1}')

        plt.figure()
        sns.heatmap(data1[0], cmap='viridis')
        plt.title(f"{i}")
        plt.savefig(f'pic/{i}.png')