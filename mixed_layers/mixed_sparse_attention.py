import math
import torch

class MixedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor, 
        sparsity_ratio: float, maintain_heads: int
    ):
        # q,k,v: [bsz, num_heads, q_len, head_dim]
        # notice forward process no need to drop heads
        bsz, num_heads, q_len, head_dim = q.shape()

        # forward: S = Q @ K.T / sqrt(d_k)
        s = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        # apply mask
        if attention_mask is not None:
            s = s + attention_mask

        # forward: softmax 
        a = torch.softmax(s, dim=-1) # [bsz, num_heads, q_len, q_len]

        # forward: O = A @ V
        o = a @ v
        o = o.transpose(1, 2).contiguous()
        o = o.reshape(bsz, q_len, num_heads * head_dim)

        # save for backward: q, k, v, a
        # firstly, compute the norm of each attention head
        norm = torch.norm(a.view(bsz, num_heads, -1), dim=-2)
        actual_maintain_heads = min(int(sparsity_ratio * num_heads), maintain_heads)
        # record the heads with minimum norm
        _, min_indices = norm.topk(actual_maintain_heads, dim=-1, largest=False)

        # save the selected heads
        q_save = torch.zeros(bsz, actual_maintain_heads, q_len, head_dim)
        k_save = torch.zeros(bsz, actual_maintain_heads, q_len, head_dim)
        v_save = torch.zeros(bsz, actual_maintain_heads, q_len, head_dim)
        a_save = torch.zeros(bsz, actual_maintain_heads, q_len, q_len)

        for i in range(bsz):
            q_save[i] = q[i, min_indices[i]]
            k_save[i] = k[i, min_indices[i]]
            v_save[i] = v[i, min_indices[i]]
            a_save[i] = a[i, min_indices[i]]

        ctx.save_for_backward(q_save, k_save, v_save, a_save)
        ctx.min_indices = min_indices
        ctx.q_shape = q.shape()
        ctx.a_shape = a.shape()
        ctx.q_device = q.device

        return o
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        min_indices = ctx.min_indices
        q_save, k_save, v_save, a_save = ctx.saved_tensors

        # convert the q, k, v, a to the original shape
        q = torch.zeros(ctx.q_shape, dtype=q_save.dtype, device=ctx.q_device)
        k = torch.zeros(ctx.q_shape, dtype=k_save.dtype, device=ctx.q_device)
        v = torch.zeros(ctx.q_shape, dtype=v_save.dtype, device=ctx.q_device)
        a = torch.zeros(ctx.a_shape, dtype=a_save.dtype, device=ctx.q_device)

        for i in range(q_save.shape[0]):
            q[i, min_indices[i]] = q_save[i]
            k[i, min_indices[i]] = k_save[i]
            v[i, min_indices[i]] = v_save[i]


        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        grad_v = a.transpose(-2, -1) @ grad_output
        grad_a = grad_output @ v.transpose(-2, -1)

        # backward of softmax
        grad_s = (grad_a - (grad_a * a).sum(dim=-1, keepdims=True)) * a

        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        grad_s = grad_s / math.sqrt(q.size(-1))
        # d L / d K = (d L / d S)^T @ Q
        grad_k = grad_s.transpose(-2, -1) @ q
        # d L / d Q = d L / d S @ K
        grad_q = grad_s @ k

        return grad_q, grad_k, grad_v, None, None, None, None, None

