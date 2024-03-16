import os
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # detect the bad channels in x
    dir = 'output'
    pt_files = [f for f in os.listdir(dir) if f.endswith('.pt') and 'lora' in f]
    for pt_file in pt_files:
        data = torch.load(f'{dir}/{pt_file}')[0]
        shape = data.shape
        record_list = []
        for i in range(shape[1]):
            max_val = max(data[:,i])
            min_val= torch.min(data[:,i])
            mean_val = torch.mean(data[:,i])
            std_val = torch.std(data[:,i])
            # record the top 16 big std channel
            record_list.append((i, std_val.item()))
            # print(f'max:{max_val} | min:{min_val} | mean:{mean_val} | std:{std_val}')
        record_list = sorted(record_list, key=lambda x: x[1], reverse=True)
        print(f'{pt_file}: {record_list[:16]}')

        # plot the 256 biggst channels:
        x = np.arange(256)
        y = np.array([item[1] for item in record_list][:256])
        plt.figure()
        plt.plot(x, y)
        plt.savefig(f'pic/statistics_{pt_file}.png')