import os
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for i in range(32):
        file = f'/home/yujin-wa20/projects/LoftQ/output/opt/base_model.model.model.decoder.layers.{i}.self_attn.gemm2_1.pt'
        data = torch.load(file).cpu().detach()
        for j in range(32):
            data1 = data[j][:, 4:]
            print(f'norm of data1 ({i} {j}): {torch.norm(data1)}')
            # plt.figure()
            # plt.imshow(data1.numpy(), cmap='viridis')
            # plt.title(f"layer {i}, head {j}")
            # plt.savefig(f'pic/opt/attention/{i}_{j}.png')

