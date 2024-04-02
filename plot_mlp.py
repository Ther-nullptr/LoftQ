import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F

if __name__ == '__main__':
    for i in range(32):
        file = f'/home/yujin-wa20/projects/LoftQ/output/prosparse_llama/base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.default.pt'
        data = (torch.load(file)[0]).cpu().detach()
        data[(data > -0.1) & (data < 0.1)] = 0

        # # plot the distribution of the data
        # plt.figure()
        # sns.histplot(data.cpu().detach().numpy().flatten(), bins=100)
        # plt.title(f"layer {i} distribution")
        # plt.savefig(f'pic/prosparse_llama/o/{i}_distribution.png')

        # # compute the sparsity ratio
        total_sparsity_ratio = torch.sum(data == 0) / data.numel()
        print(f'layer {i} zero ratio: {total_sparsity_ratio}')

        per_channel_length = data.shape[0]

        # compute the distribution of zeros in each channel
        x = np.arange(per_channel_length)
        per_channel_zero_probability = []
        for j in range(per_channel_length):
            zero_probability = np.float128(math.comb(per_channel_length, j)) * np.float128(total_sparsity_ratio ** j) * np.float128((1 - total_sparsity_ratio) ** (per_channel_length - j))
            per_channel_zero_probability.append(zero_probability)
        per_channel_zero_probability = np.array(per_channel_zero_probability)
        # print(f'layer {i} zero probability: {per_channel_zero_probability}')

        # plot the theoretical distribution
        plt.figure()
        plt.plot(x[200:], per_channel_zero_probability[200:])
        zero_counts = torch.sum(data == 0, dim = 0)
        plt.hist(zero_counts.numpy(), bins=per_channel_length, density=True)
        plt.title(f"layer {i} theoretical distribution(blue) vs practical distribution(orange)")
        plt.savefig(f'pic/prosparse_llama/o/{i}_theoretical.png')
