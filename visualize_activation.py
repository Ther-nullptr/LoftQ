import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
  datadir = '/home/yujin-wa20/projects/LoftQ/output'
  files = os.listdir(datadir)
  for file in files:
    if file.endswith('.pt') and ('softmax' in file):
      data = torch.load(os.path.join(datadir, file))
      print(data[0].shape)
      if len(data[0].shape) == 3:
        print(f'{file} has 3 tensors')
        plt.figure()
        sns.heatmap(data[0][0][:16,:16].cpu().detach(), cmap='viridis')
        plt.title(f"{file}")
        plt.savefig(f'pic/softmax_act_{file}.png')