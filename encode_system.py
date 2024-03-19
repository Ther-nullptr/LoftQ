import os
import torch
import deflate
import scipy.io
import numpy as np
import torchvision
import transformers
from typing import Optional
from dataclasses import dataclass, field

from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor
from gact.memory_efficient_function import per_block_quantization, dct_compression_for_compress, naive_quantization

# codebook
ACTAB = scipy.io.loadmat('/home/yujin-wa20/projects/jpeg-test-2/JpegCoeff.mat')['ACTAB']
DCTAB = scipy.io.loadmat('/home/yujin-wa20/projects/jpeg-test-2/JpegCoeff.mat')['DCTAB']
QTAB = scipy.io.loadmat('/home/yujin-wa20/projects/jpeg-test-2/JpegCoeff.mat')['QTAB']


@dataclass
class Arguments:
    encode_type: Optional[str] = field(
        default="DCT",
        metadata={"help": "DCT/JPEG/DIAG/DEFLATE"},
    )
    continous_zero_num: int = field(
        default=16,
        metadata={"help": "special code for continous zero num"},
    )
    compress_component: Optional[str] = field(
        default="lora",
        metadata={"help": "lora/layernorm/softmax/silu"},
    )
    quality: int = field(
        default=75,
        metadata={"help": "lora/layernorm/softmax/silu"},
    )


class Encoder():
    def __init__(
        self,
        continous_zero_num: int = 16,
        dct_zrl_code: str = '11111111001',
        diag_zrl_code: str = '11111111001',
        eob_code: str = '1010'
    ):
        self.continous_zero_num = continous_zero_num
        self.dct_zrl_code = dct_zrl_code
        self.diag_zrl_code = diag_zrl_code
        self.eob_code = eob_code

    def _preprocess_quantization(self, x, input_shape, quant_shape):
        x, quant_state = per_block_quantization(x, input_shape, quant_shape)
        return x, quant_state
    
    def preprocess_jpeg_compress(self, x, input_shape, quality, quant_shape):
        x, quant_state = self._preprocess_quantization(x, input_shape, quant_shape)
        group_shape = input_shape[:-2] + (input_shape[-2] // quant_shape, quant_shape, input_shape[-1] // quant_shape, quant_shape)
        x = x.view(group_shape).permute(0, 1, 3, 2, 4).reshape(-1, quant_shape, quant_shape)
        return x

    def preprocess_dct_compress(self, x, input_shape, quality, quant_shape):
        x, quant_state = self._preprocess_quantization(x, input_shape, quant_shape)
        dct_processor = DCTProcessor(quality=quality, interpolation=1)
        group_shape = input_shape[:-2] + (input_shape[-2] // quant_shape, quant_shape, input_shape[-1] // quant_shape, quant_shape)
        x_dct, _ = dct_compression_for_compress(x, input_shape, dct_processor, quant_shape)
        # view for easy-to-compress
        x_dct = x_dct.view(group_shape).permute(0, 1, 3, 2, 4).reshape(-1, quant_shape, quant_shape)
        return x_dct
    
    def preprocess_softmax_compress(self, x, input_shape, quality, quant_shape, pruning_val = -100):
        x = self.preprocess_jpeg_compress(x, input_shape, quality, quant_shape)
        # pruning
        x[x < -100] = -128
        return x
    
    def preprocess_deflate_compress(self, x):
        return naive_quantization(x)[0]

    def _run_size_encode(self, run, size):
        if size == 0:
            category = 0
        else:
            category = np.floor(np.log2(np.abs(size))).astype(np.int32) + 1
        idx = run * 10 + category - 1 
        code_length = ACTAB[idx][2]
        run_size_code = ACTAB[idx][3:3+code_length]
        run_size_code = ''.join([str(i) for i in run_size_code])
        return run_size_code
    
    def _binary_encode(self, data):
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
    
    def _huffman_encode(self, num):
        if num == 0:
            category = 0
        else:
            category = np.floor(np.log2(np.abs(num))).astype(np.int32) + 1
        huffman_code_length = DCTAB[category][0]
        huffman_code = DCTAB[category][1:1+huffman_code_length]
        huffman_code = ''.join([str(i) for i in huffman_code])
        return huffman_code
    
    def _generate_encode(self, size):
        idx = torch.arange(size * size).reshape(size, size)
        idx_list = []
        for i in range(0, size): # the diff of row and col
            num = size - i
            for j in range(num):
                # append the first 
                idx_list.append(idx[j, i + j].item())
            if num != size:
                for j in range(num):
                    # append the second
                    idx_list.append(idx[i + j, j].item())
        return idx_list
    
    #! notice the per process quantization shape for DCT is 64x64.
    def dct_encode(self, quantized_dct_code, quant_shape):
        # DC encode
        quantized_dct_code_dc = quantized_dct_code.numpy().astype(np.int32)[0]
        quantized_dct_code_dc_diff = -np.diff(quantized_dct_code_dc, prepend=0)
        quantized_dct_code_dc_diff[0] = quantized_dct_code_dc[0]
        dc_len = len(quantized_dct_code_dc_diff)
        dc_code = ''
        for i in range(dc_len):
            dc_code += self._huffman_encode(quantized_dct_code_dc_diff[i])
            dc_code += self._binary_encode(quantized_dct_code_dc_diff[i])

        # AC encode
        quantized_dct_code_ac = quantized_dct_code.numpy().astype(np.int32)[1:]
        ac_code_list = []
        for i in range(quant_shape):
            quantized_dct_code = quantized_dct_code_ac[:, i]
            ac_code = ''
        
            zero_num = 0
            for j in range(quant_shape - 1):
                data = quantized_dct_code[j]
                if (data != 0):
                    while (zero_num >= self.continous_zero_num):
                        ac_code += self.dct_zrl_code
                        zero_num -= self.continous_zero_num
                    ac_code += self._run_size_encode(zero_num, data)
                    ac_code += self._binary_encode(data)
                    zero_num = 0
                else:
                    zero_num += 1

            ac_code += self.eob_code
            ac_code_list.append(ac_code)

        compressed_size = len(dc_code)
        for ac_code in ac_code_list:
            compressed_size += len(ac_code)

        return compressed_size
    
    # the quantization shape is also 64x64, we directly use interface of torchvision
    def jpeg_encode(self, quantized_activation, quant_shape, quality = 75):
        quantized_activation = quantized_activation.to(torch.int16)
        quantized_activation = quantized_activation + 128
        quantized_activation_jpeg_code = torchvision.io.encode_jpeg(quantized_activation.unsqueeze(0).to(torch.uint8), quality=quality)
        compressed_size = len(quantized_activation_jpeg_code) * 8

        return compressed_size

    # the quantization shape is ususally 16 x 16
    def diagnal_encode(self, quantized_activation, quant_shape):
        idx_list = self._generate_encode(quant_shape)
        # flatten the quantized activation
        quantized_activation = quantized_activation.flatten().numpy()
        diag_code = ''
        zero_num = 0
        for i in range(quant_shape * quant_shape):
            data = quantized_activation[idx_list[i]]
            if (data != -128):
                while (zero_num >= self.continous_zero_num):
                    diag_code += self.dct_zrl_code
                    zero_num -= self.continous_zero_num
                diag_code += self._run_size_encode(zero_num, data)
                diag_code += self._binary_encode(data)
                zero_num = 0
            else:
                zero_num += 1
        
        diag_code += self.eob_code
        compressed_size = len(diag_code)
        return compressed_size
    
    def deflate_encode(self, quantized_activation, level = 6):
        # convert the activation to numpy, then to byte
        quantized_activation_numpy = quantized_activation.numpy().astype(np.int8)
        quantized_activation_code = quantized_activation_numpy.tobytes()
        compressed = deflate.gzip_compress(quantized_activation_code, level)

        return len(compressed) * 8


if __name__ == '__main__':
    parser = transformers.HfArgumentParser(Arguments)
    model_args,  = parser.parse_args_into_dataclasses()
    encoder = Encoder()

    dir = 'output'
    quality = model_args.quality
    pt_files = [f for f in os.listdir(dir) if f.endswith('.pt') and model_args.compress_component in f]

    total_before_encode_size = 0
    total_after_encode_size = 0

    for pt_file in pt_files:
        quant_shape = 16 if model_args.compress_component == 'softmax' else 64
        x = torch.load(f'{dir}/{pt_file}')
        if model_args.compress_component == 'softmax':
            # merge the 1st and 2nd dimensions
            x = torch.softmax(x, dim=-1)
            x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])
        x = x.cpu().detach()
        input_shape = x.shape
        before_encode_size = x.numel() * 8
        after_encode_size = 0
        if model_args.encode_type == 'DCT':
            x_quantized = encoder.preprocess_dct_compress(x, input_shape, quality, quant_shape)
        elif model_args.encode_type == 'JPEG':
            x_quantized = encoder.preprocess_jpeg_compress(x, input_shape, quality, quant_shape)
        elif model_args.encode_type == 'DIAG':
            x_quantized = encoder.preprocess_softmax_compress(x, input_shape, quality, quant_shape)
        elif model_args.encode_type == 'DEFLATE':
            x_quantized = encoder.preprocess_deflate_compress(x)

        for i in range(x_quantized.shape[0]):
            if model_args.encode_type == 'DCT':
                compressed_size = encoder.dct_encode(x_quantized[i], quant_shape)
            elif model_args.encode_type == 'JPEG':
                compressed_size = encoder.jpeg_encode(x_quantized[i], quant_shape, quality=quality)
            elif model_args.encode_type == 'DIAG':
                compressed_size = encoder.diagnal_encode(x_quantized[i], quant_shape)
            elif model_args.encode_type == 'DEFLATE':
                compressed_size = encoder.deflate_encode(x_quantized[i])
            after_encode_size += compressed_size

        print(f'{pt_file} ratio: {after_encode_size / before_encode_size}')

        total_before_encode_size += before_encode_size
        total_after_encode_size += after_encode_size

    print('*****************Summary*****************')
    print(f'total {model_args.compress_component} size: {total_before_encode_size}')
    print(f'total {model_args.compress_component} compress ratio: {total_after_encode_size / total_before_encode_size}')
    