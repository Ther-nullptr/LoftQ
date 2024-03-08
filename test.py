def pack_hook(x):
    print("Packing", x)
    return x

def unpack_hook(x):
    print("Unpacking", x)
    return x
a = torch.ones(5, requires_grad=True)
b = torch.ones(5, requires_grad=True) * 2

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = a * b

y.sum().backward()