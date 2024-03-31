import os
import torch
import numpy as np


if __name__ == '__main__':
    model_name = "opt"
    datadir = f'/home/yujin-wa20/projects/LoftQ/output/{model_name}'
    files = os.listdir(datadir)
    files = [file for file in files if 'input_layernorm' in file and 'layers' in file]
    # sort by the number
    files = sorted(files, key = lambda s: int(s.split('.')[5]))
    
    for i in range(0, 31):
        print(f'data {i}')
        print(files[i])
        data1 = torch.load(f'{datadir}/{files[i]}').cpu().detach().numpy() # 
        data2 = torch.load(f'{datadir}/{files[i+1]}').cpu().detach().numpy() # 

        # print the norm of data1 & diff
        norm_data1 = np.sum(np.abs(data1))
        norm_diff = np.sum(np.abs(data2 - data1))
        print(f'norm of data1: {norm_data1}')
        print(f'norm of diff: {norm_diff}')
        print(f'ratio: {norm_diff / norm_data1}')

        # plot data1, data2, data3, data2 - data1, data3 - data2 's distribution
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        sns.histplot(data1, ax=axs[0, 0])
        axs[0, 0].set_title(f"{i} data1")
        sns.histplot(data2, ax=axs[0, 1])
        axs[0, 1].set_title(f"{i} data2")
        sns.histplot(data2 - data1, ax=axs[1, 0])
        axs[1, 0].set_title(f"{i} data2 - data1")
        sns.histplot(data1, ax=axs[1, 1])
        axs[1, 1].set_title(f"{i} data1")
        sns.histplot(data2, ax=axs[1, 2])
        axs[1, 2].set_title(f"{i} data2")

        plt.savefig(f'pic/{model_name}/{files[i]}.png')