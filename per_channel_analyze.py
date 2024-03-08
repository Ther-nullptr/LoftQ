import torch

if __name__ == '__main__':
  pt_file = '/home/yujin-wa20/projects/LoftQ/output/base_model.model.model.layers.2.input_layernorm.pt'
  # 
  data = torch.load(pt_file)[0]
  shape = data.shape
  for i in range(shape[1]):
    max_val = max(data[:,i])
    min_val= torch.min(data[:,i])
    mean_val = torch.mean(data[:,i])
    std_val = torch.std(data[:,i])
    print(f'max:{max_val} | min:{min_val} | mean:{mean_val} | std:{std_val}')