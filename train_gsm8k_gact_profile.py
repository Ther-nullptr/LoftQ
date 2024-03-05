# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from enum import Enum

import gact
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import peft
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset

from gact.efficient_linear import EfficientMemoryLinear
from gact.efficient_silu import EfficientMemorySiLU
from gact.efficient_rmsnorm import EfficientMemoryRMSNorm
from gact.efficient_softmax import EfficientMemorySoftmax
from gact.efficient_dropout import EfficientMemoryDropout
from gact.utils import get_memory_usage, get_weight_memory, get_gradient_memory, get_optimizer_memory, exp_recorder
from gact.flops_ops import ModuleFLOPs_Linear, ModuleFLOPs_Norm, ModuleFLOPs_GELU, ModuleFLOPs_QK, ModuleFLOPs_OV, MethodFLOPs_softmax_from_Q

from transformers.models.mistral.modeling_mistral import MistralRMSNorm

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    full_precision:  bool = field(
        default=False,
        metadata={"help": "False: Use bitsandbytes Linear4bit, real quantization"
                          "True: Use quantization equivalent fp16/fp32 weights."
                          "Note: Set True for data parallel training"
                  },
    )
    rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    bits: int = field(
        default=4,
        metadata={"help": "Bit of the backbone. LoftQ does not require this config. Used for QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    gact: bool = field(
        default=True,
        metadata={"help": "True: Use GACT; False: Do not use GACT"},
    )
    gact_level: str = field(
        default="L1",
        metadata={"help": "GACT level."},
    )
    gradient_checkpointing_enable: bool = field(
        default=False,
        metadata={"help": "True: Use gradient checkpointing; False: Do not use gradient checkpointing"},
    )
    flash_attention: bool = field(
        default=False,
        metadata={"help": "True: Use Flash Attention; False: Do not use Flash Attention"},
    )
    get_mem: bool = field(
        default=False,
        metadata={"help": "True: Get memory usage; False: Do not get memory usage"},
    )
    get_macs: bool = field(
        default=False,
        metadata={"help": "True: Get MACs; False: Do not get MACs"},
    )
    linear_mode: str = field(
        default="NAIVE",
        metadata={"help": "Linear mode."},
    )
    linear_quality: int = field(
        default=75,
        metadata={"help": "Linear quality."},
    )
    linear_quantization_shape: int = field(
        default=16,
        metadata={"help": "Linear quantization shape."},
    )
    nonlinear_mode: str = field(
        default="NAIVE",
        metadata={"help": "NonLinear mode."},
    )
    nonlinear_quality: int = field(
        default=75,
        metadata={"help": "NonLinear quality."},
    )
    nonlinear_quantization_shape: int = field(
        default=16,
        metadata={"help": "NonLinear quantization shape."},
    )
    get_mem: bool = field(
        default=True,
        metadata={"help": "True: Get memory usage; False: Do not get memory usage"},
    )
    get_macs: bool = field(
        default=False,
        metadata={"help": "True: Get MACs; False: Do not get MACs"},
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default="gsm8k",
        metadata={"help": "Dataset name."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_value(self):
        if self.summary_type is Summary.AVERAGE:
            return self.avg
        if self.summary_type is Summary.SUM:
            return self.sum
        if self.summary_type is Summary.COUNT:
            return self.sum

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

total_flops = 0
total_flops_intensity = 0
total_forward_flops = 0
total_backward_flops = 0

class GACTTrainer(Trainer):
    def __init__(self, get_mem, get_macs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.begin = False
        self.step = 0
        self.total_mem = AverageMeter('Total Memory', ':.4e')
        self.peak_mem = AverageMeter('Peak Memory', ':.4e')
        self.activation_mem = AverageMeter('Activation Memory', ':.4e')
        self.get_mem = get_mem
        self.get_macs = get_macs
        
    def compute_loss(self, model, inputs, return_outputs=False):
        self.step += 1

        if self.get_mem and self.step > 1:
            torch.cuda.synchronize()
            # accelerator.print("===============After Data Loading=======================")
            init_mem = get_memory_usage(False)  # model size + data size
            torch.cuda.reset_peak_memory_stats()

        if self.get_macs:
            def flops_forward_hook(module, inputs, outputs):
                global total_flops, total_flops_intensity, total_forward_flops, total_backward_flops
                class_type = module.__class__.__name__
                # # only for debug
                # print(f"forward: {module.name}, {module.__class__.__name__}")

                if class_type == 'Linear': # y = x @ W + b, (B,M,K) @ (K,N) + (N,) = (B,M,N)
                    if ("base_layer" not in module.name):
                        flops = ModuleFLOPs_Linear(module, outputs, inputs[0])
                        total_flops += flops
                        total_forward_flops += flops
                        module.flops = flops # save the flops for backward

                elif class_type == 'MistralRMSNorm':
                    flops = ModuleFLOPs_Norm(module, outputs, inputs[0])
                    total_flops += flops
                    total_forward_flops += flops
                    module.flops = flops

                elif class_type == 'SiLU': # GELU, usually used in BERT type models
                    flops = ModuleFLOPs_GELU(module, outputs, inputs[0])
                    total_flops += flops
                    total_forward_flops += flops
                    module.flops = flops

                if 'q_proj' in module.name: # specially for (Q @ K.T) @ V
                    # Q @ K
                    flops = ModuleFLOPs_QK(outputs)
                    # softmax
                    flops += MethodFLOPs_softmax_from_Q(outputs)
                    # O @ V
                    flops += ModuleFLOPs_OV(outputs)
                    total_flops += flops
                    total_forward_flops += flops
                    module.attn_out_shape = outputs.shape
            
            
            def flops_backward_hook(module, grad_input, grad_output):
                global total_flops, total_flops_intensity, total_forward_flops, total_backward_flops
                class_type = module.__class__.__name__
                # # only for debug
                # print(f"backward: {module.name}, {module.__class__.__name__}")

                if class_type == 'Linear':
                    # compute gradient of activation
                    if ("base_layer" not in module.name): 
                        flops = module.flops
                        total_flops += flops
                        total_backward_flops += flops

                        # compute gradient of weight
                        if module.weight.requires_grad:
                            total_flops += flops
                            total_backward_flops += flops

                elif class_type == 'MistralRMSNorm': # TODO: norm and activation function just copy their forward, but this is not accurate!
                    flops = module.flops
                    total_flops += flops
                    total_backward_flops += flops

                elif class_type == 'GELUActivation' or class_type == 'SiLU':
                    flops = module.flops
                    total_flops += flops
                    total_backward_flops += flops

                if 'q_proj' in module.name:
                    # before: 2 (b, s, h) @ (b, h, s)
                    # after: 4 (b, s, h) @ (b, h, s) + 2 (b, s, s) @ (b, s, s)
                    attn_out_shape = module.attn_out_shape # (b, s, h)
                    b, s, h = attn_out_shape
                    flops = 2 * (4 * b * s * h * s + 2 * b * s * s * s)
                    total_flops += flops
                    total_backward_flops += flops

            # register forward hook
            for name, module in model.named_modules():
                module.name = name
                module.register_forward_hook(flops_forward_hook)
                module.register_backward_hook(flops_backward_hook)

            loss = super().compute_loss(model, inputs, return_outputs)
            loss.backward()

            print(f"Total FLOPs: {total_flops / 10**12} TFLOPs")
            print(f"Total Forward FLOPs: {total_forward_flops / 10**12} TFLOPs")
            print(f"Total Backward FLOPs: {total_backward_flops / 10**12} TFLOPs")

            exit()

        loss = super().compute_loss(model, inputs, return_outputs)
        
        if self.get_mem and self.step > 1:
            torch.cuda.synchronize()
            before_backward = get_memory_usage(True)
        
            loss.backward()
            torch.utils.checkpoint.first_iter = False

            torch.cuda.synchronize()
            for t in inputs:
                del t
            after_backward = get_memory_usage(False)

            self.total_mem.update(before_backward)
            self.activation_mem.update(before_backward - after_backward)
            self.peak_mem.update(
                torch.cuda.max_memory_allocated()
            )

            print("peak %d MB" % (self.peak_mem.get_value() / 1024 / 1024))
            print("total %d MB" % (self.total_mem.get_value() / 1024 / 1024))
            print("activation %d MB" % (self.activation_mem.get_value() / 1024 / 1024))
            
            print("weight %d MB" % (get_weight_memory(model) / 1024 / 1024))
            print("gradient %d MB" % (get_gradient_memory(model) / 1024 / 1024))
            print("optimizer %d MB" % (get_optimizer_memory(self.optimizer) / 1024 / 1024))
            print('-------------------------------------------------')
            exit(0)

        return loss


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        #! a tricky way to pad the input_ids and labels to 16's multiple
        # 1. find the max length of input_ids
        max_len = max([len(input_id) for input_id in input_ids])
        # 2. pad the input_ids and labels to 16's multiple
        max_len = (max_len + 63) // 64 * 64
        # 3. generate a max_len tensor
        max_len_tensor = torch.randn(max_len).to(torch.int64)
        # 4. append the max_len tensor to the input_ids and labels
        input_ids.append(max_len_tensor)
        labels.append(max_len_tensor)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        # delete the max_len tensor
        input_ids = input_ids[:-1]
        labels = labels[:-1]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    train_set = dataset['train']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def replace_module(module, compress_config):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and (child.weight.requires_grad) and (name != 'class_intermediate' and name != 'out_proj' and child.in_features > 100):
            original_weight_data = child.weight.data
            original_bias_data = child.bias.data if child.bias is not None else None
            new_child = EfficientMemoryLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                compress_type=compress_config['linear']['mode'],
                compress_quality=compress_config['linear']['quality'],
            )
            new_child.weight.data = original_weight_data
            if child.bias is not None:
                new_child.bias.data = original_bias_data
            setattr(module, name, new_child)
        elif isinstance(child, torch.nn.SiLU):
            setattr(module, name, EfficientMemorySiLU(compress_type=compress_config['nonlinear']['mode'], compress_quality=compress_config['nonlinear']['quality'], quantization_shape=compress_config['nonlinear']['quantization_shape']))
        elif isinstance(child, MistralRMSNorm):
            original_weight_data = child.weight.data
            new_child = EfficientMemoryRMSNorm(
                normalized_shape=child.weight.data.shape,
                eps=child.variance_epsilon,
                elementwise_affine=True,
                bias=False,
                compress_type=compress_config['nonlinear']['mode'],
                compress_quality=compress_config['nonlinear']['quality'],
                quantization_shape=compress_config['nonlinear']['quantization_shape']
            )
            new_child.weight.data = original_weight_data
            setattr(module, name, new_child)
        elif isinstance(child, torch.nn.Softmax):
            setattr(module, name, EfficientMemorySoftmax(-1, compress_type=compress_config['nonlinear']['mode'], compress_quality=compress_config['nonlinear']['quality'], quantization_shape=compress_config['nonlinear']['quantization_shape']))
        elif isinstance(child, torch.nn.Dropout):
            setattr(module, name, EfficientMemoryDropout(child.p))
        else:
            replace_module(child, compress_config)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compress_config = {
        'linear': {
            'mode': model_args.linear_mode,
            'quality': model_args.linear_quality
        },
        'nonlinear': {
            'mode': model_args.nonlinear_mode,
            'quality': model_args.nonlinear_quality,
            'quantization_shape': model_args.nonlinear_quantization_shape
        },
    }

    if model_args.full_precision:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            attn_implementation="flash_attention_2" if model_args.flash_attention else "eager",
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            use_cache=False if model_args.gradient_checkpointing_enable else True,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            attn_implementation="flash_attention_2" if model_args.flash_attention else "eager",
        )
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=model_args.gradient_checkpointing_enable, gradient_checkpointing_kwargs={"use_reentrant": False})
        if (model_args.gradient_checkpointing_enable):
            print("Gradient Checkpointing is enabled")
    
    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model,
                                          model_args.adapter_name_or_path,
                                          is_trainable=True,
                                          token=model_args.token,
                                          )
    else:
        model = PeftModel.from_pretrained(model,
                                          model_args.model_name_or_path,
                                          subfolder='loftq_init',
                                          is_trainable=True,
                                          token=model_args.token,
                                          )
        
    # replace the module
    replace_module(model, compress_config)
    print(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )
    trainer = GACTTrainer(get_mem=model_args.get_mem, get_macs=model_args.get_macs, model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()


if __name__ == "__main__":
    train()
