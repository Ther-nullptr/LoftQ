import math
import torch

class GatedMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b, activation_forward, activation_backward):
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
        if activation_backward == 'relu':
            mask = y1 < 0
            if activation_forward != 'relu':
                xL = torch.relu(xL)
            ctx.save_for_backward(x1, y1_lora_a, mask, xL, xR, y2_lora_a, x3, y3_lora_a, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b)
        else:
            ctx.save_for_backward(x1, y1_lora_a, y1, xL, xR, y2_lora_a, x3, y3_lora_a, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b)

        return y3
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.activation_backward == 'relu':
            x1, y1_lora_a, mask, xL, xR, y2_lora_a, x3, y3_lora_a, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b = ctx.saved_tensors
        else:
            x1, y1_lora_a, y1, xL, xR, y2_lora_a, x3, y3_lora_a, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b = ctx.saved_tensors

        # down proj part
        # d L / d w_down_lora_a = x2.T @ d L / d y2 @ w_down_lora_b.T
        # TODO: x2 maybe sparse
        grad_w_down_lora_a = x3.T @ (grad_output @ w_down_lora_b.T)
        # d L / d w_down_lora_b = y2_lora_a.T @ d L / d y2
        grad_w_down_lora_b = y3_lora_a.T @ grad_output
        # d L / d x2 = d L / d y2 @ w_down.T + d L / d y2 @ w_down_lora_b.T @ w_down_lora_a.T
        grad_x2 = grad_output @ w_down.T + grad_output @ w_down_lora_b.T @ w_down_lora_a.T

        # hadamard product
        # d L / d xL = d L / d x3 * xR
        grad_xL = grad_x2 * xR
        # d L / d xR = d L / d x3 * xL
        grad_xR = grad_x2 * xL

        # activation part
        if ctx.activation_backward == 'relu':
            grad_y1 = grad_xL.clone()
            grad_y1[mask] = 0
        elif ctx.activation_backward == 'silu':
            sigmoid = torch.sigmoid(y1)
            grad_y1 = sigmoid * (1 + y1 - y1 * sigmoid) * grad_xL
        elif ctx.activation_backward == 'gelu':
            gamma = math.sqrt(2 / math.pi)
            kappa = 0.044715
            grad_y1 = gamma * (y1 + kappa * y1 ** 3)
            tanh_y = torch.tanh(y1)
            grad_y1 = 0.5 * ((1 + tanh_y) + y1 * ((1 - tanh_y ** 2) * gamma * (1 + 3 * kappa * y1 ** 2))) * grad_xL

        # up proj part
        # d L / d w_up_lora_a = x1.T @ d L / d xL @ w_up_lora_b.T
        grad_w_up_lora_a = xR.T @ (grad_xL @ w_up_lora_b.T)
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

        return grad_x1, None, None, grad_w_gate_lora_a, grad_w_gate_lora_b, None, None, grad_w_up_lora_a, grad_w_up_lora_b, None, None, grad_w_down_lora_a, grad_w_down_lora_b, None, None
    

class GatedMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, rank, activation_forward='relu', activation_backward='relu', dropping=0.5, bias=False):
        super(GatedMLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropping = dropping
        self.rank = rank

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
        self.activation_backward = activation_backward

    def forward(self, input):
        return GatedMLPFunc.apply(
            input,
            self.gate_proj.weight,
            self.gate_proj.bias,
            self.gate_proj_lora_a.weight,
            self.gate_proj_lora_b.weight,
            self.up_proj.weight,
            self.up_proj.bias,
            self.up_proj_lora_a.weight,
            self.up_proj_lora_b.weight,
            self.down_proj.weight,
            self.down_proj.bias,
            self.down_proj_lora_a.weight,
            self.down_proj_lora_b.weight,
            self.activation_forward,
            self.activation_backward
        )