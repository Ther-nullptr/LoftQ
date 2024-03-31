import cv2
import numpy as np
import matplotlib.pyplot as plt
from gact.utils import get_dct_matrix

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
  original_data = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
  original_data_col, original_data_row = original_data.shape

  # construct a table to record every channel's value in the DCT matrix
  channel_table = np.zeros((original_data_col // 64, original_data_row, 64))

  D = get_dct_matrix(64)
  # compute every chunk's DCT
  for i in range(0, original_data_col, 64):
    for j in range(0, original_data_row):
      chunk = original_data[i:i+64, j:j+1]
      C = np.dot(D, chunk)
      C = np.round(C).flatten()
      # no quantization!!!

      # return the zigzag order of the DCT matrix
      channel_table[i // 64][j] = C

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
  plt.bar(x, per_channel_entropy)
  plt.xlabel('Channel')
  plt.ylabel('Entropy')
  plt.title('Entropy (Lena)')
  plt.savefig('entropy.png')
