import torch
from safetensors.torch import load_file
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
  for i in range(32):
    # load input
    input_activation = torch.load(f'/home/yujin-wa20/projects/LoftQ/output/base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.default.pt').cuda().to(torch.float32)
    main_weight = torch.load('/home/yujin-wa20/projects/aliendao/dataroot/models/mistralai/Mistral-7B-v0.1/pytorch_model-00001-of-00002.bin')
    main_weight = main_weight[f'model.layers.{i}.self_attn.k_proj.weight'].cuda().to(torch.float32)
    lora_weight = load_file('/home/yujin-wa20/projects/LoftQ/exp_results/gsm8k_mistral_7b_4bit_64rank_loftq_None/gsm8k_mistral_7b_4bit_64rank_loftq_None/Mistral-7B-v0.1-4bit-16rank/ep_6/lr_0.0003/seed_11/adapter_model.safetensors')
    lora_a = lora_weight[f'base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight'].cuda().to(torch.float32)
    lora_b = lora_weight[f'base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.weight'].cuda().to(torch.float32)
    # perform operation
    main_output = torch.matmul(input_activation, main_weight.T)
    lora_output = input_activation @ lora_a.T @ lora_b.T
    # plot the data
    plt.figure()
    sns.heatmap(main_output[0].cpu().detach(), cmap='viridis')
    plt.title(f"main_output")
    plt.savefig(f'pic/main_output_{i}.png')
    plt.figure()
    sns.heatmap(lora_output[0].cpu().detach(), cmap='viridis')
    plt.title(f"lora_output")
    plt.savefig(f'pic/lora_output_{i}.png')
    # save output