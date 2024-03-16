import torch
import bitsandbytes as bnb

if __name__ == '__main__':
    # i = 1
    # main_weight = torch.load('/home/yujin-wa20/projects/aliendao/dataroot/models/mistralai/Mistral-7B-v0.1/pytorch_model-00001-of-00002.bin')
    # w = main_weight[f'model.layers.{i}.self_attn.k_proj.weight'].cuda().to(torch.float32)

    # w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type='nf4')
    # w_dequantize =  bnb.functional.dequantize_4bit(w_4bit, quant_state)

    # # print the average mse of w_4bit and w_dequantize
    # print(torch.mean(torch.abs(w_dequantize - w)))


    # detect the bad channels in x
    pt_file = '/home/yujin-wa20/projects/LoftQ/output/base_model.model.model.layers.2.input_layernorm.pt'
    # 
    data = torch.load(pt_file)[0]
    shape = data.shape
    record_list = []
    for i in range(shape[1]):
        max_val = max(data[:,i])
        min_val= torch.min(data[:,i])
        mean_val = torch.mean(data[:,i])
        std_val = torch.std(data[:,i])
        # record the top 16 big std channel
        record_list.append((i, torch.abs(mean_val)))
        # print(f'max:{max_val} | min:{min_val} | mean:{mean_val} | std:{std_val}')
    record_list = sorted(record_list, key=lambda x: x[1], reverse=True)
    print(record_list[:16])
