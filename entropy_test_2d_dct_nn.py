import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gact.utils import get_dct_matrix
from gact.memory_efficient_function import per_block_quantization

def naive_adjustment(x, input_shape, quantization_shape = 64):
  group_size_1 = input_shape[-2] // quantization_shape
  group_size_2 = input_shape[-1] // quantization_shape
  x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
  x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
  x = x.reshape(input_shape)
  return x

def preprocess_quantization(x, input_shape, quant_shape):
  x, quant_state = per_block_quantization(x, input_shape, quant_shape)
  return x, quant_state

def zigzag(matrix):
  zigzag_matrix = []
  for i in range(0, 15):
    for j in range(8):
      if i - j >= 0 and i - j < 8:
        if i % 2 == 0:
          zigzag_matrix.append(matrix[i - j][j])
        else:
          zigzag_matrix.append(matrix[j][i - j])
  return np.array(zigzag_matrix)

def shannon_entropy(vector):
  # 统计向量中每个值的出现次数
  unique, counts = np.unique(vector, return_counts=True)
  # 计算概率分布
  probabilities = counts / len(vector)
  # 计算香农信息熵
  entropy = -np.sum(probabilities * np.log2(probabilities))
  return entropy

if __name__ == '__main__':
  original_data = torch.load('/home/yujin-wa20/projects/LoftQ/output/mistral/base_model.model.model.layers.15.self_attn.k_proj.lora_A.default.pt')[0]
  original_data, _ = preprocess_quantization(original_data, original_data.shape, 64)
  original_data = naive_adjustment(original_data, original_data.shape, 64)
  original_data = original_data.cpu().detach().numpy()
  original_data_col, original_data_row = original_data.shape

  # plot the original data
  plt.figure()
  sns.heatmap(original_data)
  plt.savefig('original_data.png')

  # construct a table to record every channel's value in the DCT matrix
  channel_table = np.zeros((original_data_col // 8, original_data_row // 8, 64))
  D = get_dct_matrix(8)
  # compute every chunk's DCT
  for i in range(0, original_data_col, 8):
    for j in range(0, original_data_row, 8):
      chunk = original_data[i:i+8, j:j+8]
      
      C = np.dot(np.dot(D, chunk), D.T)
      C = np.round(C)

      # return the zigzag order of the DCT matrix
      zigzag_matrix = zigzag(C)
      channel_table[i // 8][j // 8] = zigzag_matrix

  per_channel_entropy = []
  for i in range(64):
    # get every DCT channel's cross entropy
    channel = channel_table[:, :, i]
    channel = channel.flatten()
    entropy = shannon_entropy(channel)
    print(f'Channel {i}\'s entropy: {entropy}')
    per_channel_entropy.append(entropy)

  # plot the entropy of every channel
  x = np.arange(64)
  plt.figure()
  plt.bar(x, per_channel_entropy)
  plt.xlabel('Channel')
  plt.ylabel('Entropy')
  plt.title('Entropy (V layer feature)')
  plt.savefig('entropy.png')
