import os
import torch
import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor
import torchvision
from gact.memory_efficient_function import per_block_quantization, jpeg_compression_cpu

ZRL = '11111111001'
EOB = '1010'

ACTAB = scipy.io.loadmat('/home/yujin-wa20/projects/jpeg-test-2/JpegCoeff.mat')['ACTAB']
DCTAB = scipy.io.loadmat('/home/yujin-wa20/projects/jpeg-test-2/JpegCoeff.mat')['DCTAB']
QTAB = scipy.io.loadmat('/home/yujin-wa20/projects/jpeg-test-2/JpegCoeff.mat')['QTAB']

def run_size_encode(run, size):
  if size == 0:
    category = 0
  else:
    category = np.floor(np.log2(np.abs(size))).astype(np.int32) + 1
  idx = run * 10 + category - 1 
  code_length = ACTAB[idx][2]
  run_size_code = ACTAB[idx][3:3+code_length]
  run_size_code = ''.join([str(i) for i in run_size_code])
  return run_size_code


def binary_encode(data):
  if data == 0:
    category_code = ''
  else:
    category_code = bin(np.abs(data))[2:]
    category_code = [int(bit) for bit in category_code]
    if data < 0:
      for i in range(len(category_code)):
        category_code[i] = 1 - category_code[i]
    category_code = ''.join([str(i) for i in category_code])
  return category_code


def huffman_encode(num):
  if num == 0:
    category = 0
  else:
    category = np.floor(np.log2(np.abs(num))).astype(np.int32) + 1
  huffman_code_length = DCTAB[category][0]
  huffman_code = DCTAB[category][1:1+huffman_code_length]
  huffman_code = ''.join([str(i) for i in huffman_code])
  return huffman_code


def encode(quantized_dct_code, quant_shape):
  quantized_dct_code_dc = quantized_dct_code.numpy().astype(np.int32)[0]
  quantized_dct_code_dc_diff = -np.diff(quantized_dct_code_dc, prepend=0)
  quantized_dct_code_dc_diff[0] = quantized_dct_code_dc[0]
  dc_len = len(quantized_dct_code_dc_diff)

  dc_code = ''
  for i in range(dc_len):
    dc_code += huffman_encode(quantized_dct_code_dc_diff[i])
    dc_code += binary_encode(quantized_dct_code_dc_diff[i])

  # AC encode
  quantized_dct_code_ac = quantized_dct_code.numpy().astype(np.int32)[1:]
  ac_code_list = []
  for i in range(quant_shape):
    quantized_dct_code = quantized_dct_code_ac[:, i]
    ac_code = ''
  
    zero_num = 0
    for i in range(quant_shape-1):
      data = quantized_dct_code[i]
      if (data != 0):
        while (zero_num >= 16):
          ac_code += ZRL
          zero_num -= 16
        ac_code += run_size_encode(zero_num, data)
        ac_code += binary_encode(data)
        zero_num = 0
      else:
        zero_num += 1

    ac_code += EOB
    ac_code_list.append(ac_code)

  original_size = quant_shape * quant_shape * 8
  compressed_size = len(dc_code)
  for ac_code in ac_code_list:
    compressed_size += len(ac_code)

  return compressed_size


if __name__ == '__main__':
  quality = 75
  dir = 'output'
  pt_files = [f for f in os.listdir(dir) if f.endswith('.pt') and 'lora' in f]
  for pt_file in pt_files:
    print(pt_file)
    quant_shape = 64
    tensor = torch.load(f'{dir}/{pt_file}').cpu().detach()
    
    input_shape = tensor.shape
    x, quant_state = per_block_quantization(tensor, input_shape, quant_shape)
    # split the last dimension into 64
    group_shape = input_shape[:-2] + (input_shape[-2] // quant_shape, quant_shape, input_shape[-1] // quant_shape, quant_shape)

    jpeg_processor = JPEGProcessor(quality=quality)

    original_x = x
    x = jpeg_compression_cpu(x, input_shape, jpeg_processor, quant_shape)

    x = x.view(group_shape).permute(0, 1, 3, 2, 4).reshape(-1, quant_shape, quant_shape)
    original_x = original_x.view(group_shape).view(-1, 64, 64)

    total_mse = 0

    for i in range(x.shape[0]):
      plt.figure()
      sns.heatmap(original_x[i], cmap='viridis')
      plt.title(f'Original_{i}')
      plt.savefig(f'pic/{pt_file}_{i}.png')

      plt.figure()
      sns.heatmap(x[i], cmap='viridis')
      plt.title(f'Compressed_{i}')
      plt.savefig(f'pic/{pt_file}_{i}_compressed.png')

      plt.figure()
      plt.hist(original_x[i])
      plt.title(f'Hist_{i}')
      plt.savefig(f'pic/{pt_file}_{i}_hist.png')

      # compute per chunks's PSNR
      mse = torch.mean((x[i].to(torch.float32) - original_x[i].to(torch.float32))**2)
      total_mse += mse

    psnr = 10 * torch.log10(255**2 / (total_mse / x.shape[0]))
    print(psnr)
      

    
    