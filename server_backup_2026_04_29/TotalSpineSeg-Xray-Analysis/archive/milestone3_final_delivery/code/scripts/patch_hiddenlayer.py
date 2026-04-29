import torch

# Monkey-patch for hiddenlayer compatibility with modern PyTorch
if not hasattr(torch.onnx, '_optimize_graph'):
    torch.onnx._optimize_graph = lambda *args, **kwargs: None
    print("Monkey-patched torch.onnx._optimize_graph for hiddenlayer compatibility.")
