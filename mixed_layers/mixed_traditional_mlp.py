import torch

class MixedTraditionalMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class MixedTraditionalMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=True):
        pass