import os
import torch
import numpy as np


if __name__ == '__main__':
    model_name = "mistral"
    datadir = f'/home/yujin-wa20/projects/LoftQ/output/{model_name}'
    files = os.listdir(datadir)
    files_1 = [file for file in files if 'input_layernorm' in file and 'layers' in file]
    files_1 = sorted(files_1, key = lambda s: int(s.split('.')[4]))

    files_2 = [file for file in files if 'post_attention_layernorm' in file and 'layers' in file]
    files_2 = sorted(files_2, key = lambda s: int(s.split('.')[4]))
    
    for i in range(0, 31):
        print(f'data {i}')
        data1 = torch.load(f'{datadir}/{files_1[i]}').cpu().detach().numpy()[0]
        data2 = torch.load(f'{datadir}/{files_2[i]}').cpu().detach().numpy()[0]
        data3 = torch.load(f'{datadir}/{files_1[i+1]}').cpu().detach().numpy()[0]

        # print the norm of data1 & diff
        norm_data1 = np.sum(np.abs(data1))
        norm_data2 = np.sum(np.abs(data2))
        norm_diff1 = np.sum(np.abs(data2 - data1))
        norm_diff2 = np.sum(np.abs(data3 - data2))
        print(f'norm of data1: {norm_data1}')
        print(f'norm of diff1: {norm_diff1}')
        print(f'norm of diff2: {norm_diff2}')
        print(f'ratio1: {norm_diff1 / norm_data1}')
        print(f'ratio2: {norm_diff2 / norm_data2}')

        # plot data1, data2, data3, data2 - data1, data3 - data2 in one figure
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        sns.heatmap(data1, ax=axs[0, 0])
        axs[0, 0].set_title(f"{i} data1")
        sns.heatmap(data2, ax=axs[0, 1])
        axs[0, 1].set_title(f"{i} data2")
        sns.heatmap(data3, ax=axs[1, 0])
        axs[1, 0].set_title(f"{i} data3")
        sns.heatmap(data2 - data1, ax=axs[1, 1])
        axs[1, 1].set_title(f"{i} data2 - data1")
        sns.heatmap(data3 - data2, ax=axs[1, 2])
        axs[1, 2].set_title(f"{i} data3 - data2")

        plt.savefig(f'pic/{model_name}/{files_1[i]}.png')
        # sparsity_ratio = np.count_nonzero(data2 - data1) / data1.size
        # print(f'sparsity ratio: {1 - sparsity_ratio}')

        # also plot the distribution
