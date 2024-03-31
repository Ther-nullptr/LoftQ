import torch
from safetensors.torch import load_file
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
  for i in range(23, 32):
    print(f'layer {i}')
    # load input
    input_activation = torch.load(f'/home/yujin-wa20/projects/LoftQ/output/mistral/base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.default.pt').cuda().to(torch.float32)
    main_weight = torch.load('/home/yujin-wa20/projects/aliendao/dataroot/models/mistralai/Mistral-7B-v0.1/pytorch_model-00002-of-00002.bin')
    main_weight = main_weight[f'model.layers.{i}.self_attn.o_proj.weight'].cuda().to(torch.float32)
    lora_weight = load_file('/home/yujin-wa20/projects/LoftQ/exp_results/gsm8k_mistral_7b_4bit_64rank_loftq_None/gsm8k_mistral_7b_4bit_64rank_loftq_None/Mistral-7B-v0.1-4bit-16rank/ep_6/lr_0.0003/seed_11/adapter_model.safetensors')
    lora_a = lora_weight[f'base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.weight'].cuda().to(torch.float32)
    lora_b = lora_weight[f'base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.weight'].cuda().to(torch.float32)
    # perform operation
    main_output = torch.matmul(input_activation, main_weight.T)
    lora_output = input_activation @ lora_a.T @ lora_b.T
    # [bsz, seq_len, hidden_size] -> [bsz, n_heads, seq_len, head_dim]
    bsz, seq_len, hidden_size = main_output.shape
    print(f'bsz: {bsz}, seq_len: {seq_len}, hidden_size: {hidden_size}')
    head_dim = 128
    n_heads = hidden_size // head_dim
    main_output = main_output.reshape(bsz, seq_len, n_heads, head_dim)[0].transpose(0, 1)
    main_output = main_output.reshape(n_heads, -1)
    lora_output = lora_output.reshape(bsz, seq_len, n_heads, head_dim)[0].transpose(0, 1)
    lora_output = lora_output.reshape(n_heads, -1)

    # print per head norm(dim=1)
    norm_main_output = torch.norm(main_output, dim=1)
    print(f'norm of main_output: {norm_main_output.shape}')
    norm_lora_output = torch.norm(lora_output, dim=1)
    print(f'norm of lora_output: {norm_lora_output.shape}')

    print(f'norm of main_output: {norm_main_output}')
    print(f'norm of lora_output: {norm_lora_output}')

    # # # plot the data
    # plt.figure()
    # plt.hist(main_output[0].cpu().detach().numpy(), bins = 100)
    # plt.title(f"main_output")
    # plt.savefig(f'pic/main_output_hist_{i}.png')
    # plt.figure()
    # plt.hist(lora_output[0].cpu().detach().numpy(), bins = 100)
    # plt.title(f"lora_output")
    # plt.savefig(f'pic/lora_output_hist_{i}.png')