import math
import torch

class MixedSparseGatedMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b, activation_forward, sparsity_ratio, maintain_channels):
        # forward process: gate_proj
        y1_main = x1 @ w_gate + b_gate if b_gate is not None else x1 @ w_gate
        y1_lora_a = x1 @ w_gate_lora_a
        y1_lora = y1_lora_a @ w_gate_lora_b
        y1 = y1_main + y1_lora

        # forward process: up_proj
        y2_main = x1 @ w_up + b_up if b_up is not None else x1 @ w_up
        y2_lora_a = x1 @ w_up_lora_a
        y2_lora = y2_lora_a @ w_up_lora_b
        y2 = y2_main + y2_lora

        # apply activation function
        if activation_forward == 'relu':
            xL = torch.relu(y1)
        elif activation_forward == 'silu':
            xL = torch.silu(y1)
        elif activation_forward == 'gelu':
            xL = torch.gelu(y1)

        # hadamard product
        xR = y2
        x3 = xL * xR

        # forward process: down_proj
        y3_main = x3 @ w_down + b_down if b_down is not None else x3 @ w_down
        y3_lora_a = x3 @ w_down_lora_a
        y3_lora = y3_lora_a @ w_down_lora_b
        y3 = y3_main + y3_lora

        # save: x1, y1_lora_a, y1(for soft activations), mask(for hard activations), x2, y2_lora_a
        mask = y1 < 0
        if activation_forward != 'relu':
            xL = torch.relu(xL)

        # the pruning of x3 etc. is not urgent, we can implement it in other place
        zero_count_per_channel = (x3 == 0).sum(dim=-2) # [bs, seq_len, hidden_dim] -> [bs, hidden_dim]
        actual_maintain_channel = min(int(sparsity_ratio * x3.size(-1)), maintain_channels)
        # record the top sparsity_ratio channels
        _, topk_indices = zero_count_per_channel.topk(actual_maintain_channel, dim=-1, largest=False)
        # delete the sparse channels, and also delete the corresponding x3 channels
        x3 = x3[:, :, topk_indices]
        xL = xL[:, :, topk_indices]
        xR = xR * mask # the xR is sparse version for storage
        xR = xR[:, :, topk_indices]

        ctx.save_for_backward(x1, y1_lora_a, mask, xL, xR, y2_lora_a, x3, y3_lora_a, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b)
        ctx.topk_indices = topk_indices
        ctx.x3_shape = x3.shape

        return y3
    
    @staticmethod
    def backward(ctx, grad_output):
        x1, y1_lora_a, mask, xL, xR, y2_lora_a, x3, y3_lora_a, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b = ctx.saved_tensors

        # convert the x3, xL, xR to the original shape
        x3 = torch.zeros(ctx.x3_shape, device=x3.device).scatter(-1, ctx.topk_indices.unsqueeze(1).expand(-1, x3.size(1), -1), x3)
        xL = torch.zeros(ctx.x3_shape, device=x3.device).scatter(-1, ctx.topk_indices.unsqueeze(1).expand(-1, x3.size(1), -1), xL)
        xR = torch.zeros(ctx.x3_shape, device=x3.device).scatter(-1, ctx.topk_indices.unsqueeze(1).expand(-1, x3.size(1), -1), xR)

        # down proj part
        # d L / d w_down_lora_a = x3.T @ d L / d y3 @ w_down_lora_b.T
        # TODO: x2 maybe sparse
        grad_w_down_lora_a = x3.T @ (grad_output @ w_down_lora_b.T)
        # d L / d w_down_lora_b = y3_lora_a.T @ d L / d y3
        grad_w_down_lora_b = y3_lora_a.T @ grad_output
        # d L / d x3 = d L / d y3 @ w_down.T + d L / d y3 @ w_down_lora_b.T @ w_down_lora_a.T
        grad_x3 = grad_output @ w_down.T + grad_output @ w_down_lora_b.T @ w_down_lora_a.T

        # hadamard product
        # # d L / d xL = d L / d x3 * xR
        # grad_xL = grad_x3 * xR
        # d L / d xR = d L / d x3 * xL
        grad_xR = grad_x3 * xL
        # activation part
        grad_y1 = grad_x3 * xR # notice that the xR there is not the original xR, but the sparse version (xR * mask)

        # up proj part
        # d L / d w_up_lora_a = x1.T @ d L / d xL @ w_up_lora_b.T
        grad_w_up_lora_a = xR.T @ (grad_xR @ w_up_lora_b.T)
        # d L / d w_up_lora_b = y1_lora_a.T @ d L / d xL
        grad_w_up_lora_b = y2_lora_a.T @ grad_xR
        # d L / d x1 = d L / d xR @ w_up.T + d L / d xR @ w_up_lora_b.T @ w_up_lora_a.T
        grad_x1 = grad_xR @ w_up.T + grad_xR @ w_up_lora_b.T @ w_up_lora_a.T

        # gate proj part
        # d L / d w_gate_lora_a = x1.T @ d L / d xL @ w_gate_lora_b.T
        grad_w_gate_lora_a = x1.T @ (grad_y1 @ w_gate_lora_b.T)
        # d L / d w_gate_lora_b = y1_lora_a.T @ d L / d xL
        grad_w_gate_lora_b = y1_lora_a.T @ grad_y1
        # d L / d x1 = d L / d xL @ w_gate.T + d L / d xL @ w_gate_lora_b.T @ w_gate_lora_a.T
        grad_x1 += grad_y1 @ w_gate.T + grad_y1 @ w_gate_lora_b.T @ w_gate_lora_a.T

        return grad_x1, None, None, grad_w_gate_lora_a, grad_w_gate_lora_b, None, None, grad_w_up_lora_a, grad_w_up_lora_b, None, None, grad_w_down_lora_a, grad_w_down_lora_b, None, None, None
    

class MixedSparseGatedMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, rank, activation_forward='relu', sparsity_ratio=0.5, maintain_channels=10, bias=False):
        super(MixedSparseGatedMLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rank = rank
        self.sparsity_ratio = sparsity_ratio
        self.maintain_channels = maintain_channels

        # linear layers
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.gate_proj_lora_a = torch.nn.Linear(self.hidden_size, self.rank, bias=False)
        self.gate_proj_lora_b = torch.nn.Linear(self.rank, self.intermediate_size, bias=False)

        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj_lora_a = torch.nn.Linear(self.hidden_size, self.rank, bias=False)
        self.up_proj_lora_b = torch.nn.Linear(self.rank, self.intermediate_size, bias=False)

        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.down_proj_lora_a = torch.nn.Linear(self.intermediate_size, self.rank, bias=False)
        self.down_proj_lora_b = torch.nn.Linear(self.rank, self.hidden_size, bias=False)

        self.activation_forward = activation_forward

    def forward(self, input):
        return MixedSparseGatedMLPFunc.apply(
            input,
            self.gate_proj.weight.T,
            self.gate_proj.bias,
            self.gate_proj_lora_a.weight.T,
            self.gate_proj_lora_b.weight.T,
            self.up_proj.weight.T,
            self.up_proj.bias,
            self.up_proj_lora_a.weight.T,
            self.up_proj_lora_b.weight.T,
            self.down_proj.weight.T,
            self.down_proj.bias,
            self.down_proj_lora_a.weight.T,
            self.down_proj_lora_b.weight.T,
            self.activation_forward,
            self.sparsity_ratio,
            self.maintain_channels
        )