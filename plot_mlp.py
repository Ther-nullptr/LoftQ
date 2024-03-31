import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F

if __name__ == '__main__':
    for i in range(12):
        file = f'/home/yujin-wa20/projects/GACT-ICML/benchmark/text_classification/output/base_model.model.roberta.encoder.layer.{i}.intermediate.intermediate_act_fn.pt'
        data = (torch.load(file)[0]).cpu().detach()

        plt.figure()
        sns.heatmap((data > -1) & (data < 0), cmap='viridis')
        plt.title(f"layer {i}")
        plt.savefig(f'pic/BERT/{i}.png')

        # plt.figure()
        # sns.histplot(data.numpy().flatten(), bins=100, legend=False)
        # plt.title(f"layer {i}")
        # plt.savefig(f'pic/BERT/{i}.png')

        # print(f'layer {i} zero ratio: {torch.sum(data == 0) / data.numel()}')


        # # # plot the heatmap
        # # plt.figure()
        # # sns.histplot(data.numpy(), bins=100, kde=True)
        # # plt.title(f"layer {i}")
        # # plt.savefig(f'pic/mistral/lora_A/{i}.png')
        # # print(f'save to pic/mistral/lora_A/{i}.png')

        # # # data = data

        # # print the zero counts of a data
        # print(f'layer {i} zero ratio: {torch.sum(data == 0) / data.numel()}')
        # print(data.shape)
        
        # zero_counts = torch.sum(data == 0, dim = 0)

        # # # plot the distribution of data
        # plt.figure()
        # plt.hist(zero_counts.numpy(), bins=100)

        # # sns.histplot(zero_counts.numpy(), bins=100, kde=True)
        # # plt.title(f"layer {i}")

        # # plt.savefig(f'pic/prosparse_llama/hadamard_1/{i}.png')
        # # # print('hist done')

        # percentiles = np.percentile(zero_counts.numpy(), [25, 50, 75])  # 计算第25、50和75百分位数
        # for percentile in percentiles:
        #     plt.axvline(percentile, color='r', linestyle='dashed', linewidth=1)
        #     plt.text(percentile, 50, f'{percentile:.2f}', rotation=90, verticalalignment='bottom', color='r')

        # # plt.title(f"layer {i}")
        # plt.savefig(f'pic/BERT/{i}.png')