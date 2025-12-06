import torch
import torch_webgpu
from torch_webgpu import webgpu_backend


@torch.compile(backend=webgpu_backend)
def fn(x):
    return x + x


if __name__ == "__main__":
    a = torch.tensor([2, 3, 4], device="webgpu")
    out = fn(a)
    out = out.to("cpu")
    print(out)
