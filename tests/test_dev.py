import torch
import torch_webgpu
from torch_webgpu import webgpu_backend


@torch.compile(backend=webgpu_backend)
def fn(x):
    a = torch.tensor([-1.5, 2.7, 1.0, 2.0], device="webgpu")
    b = torch.tensor([-1.0, 0.9, 1.1, -2.1], device="webgpu")
    result = a + b
    result = torch.relu(result)
    result = result.to("cpu")
    return result


if __name__ == "__main__":
    result = fn(torch.empty(0))
    expected = torch.tensor([0, 3.6, 2.1, 0], device="cpu")
    assert torch.allclose(result, expected)
    print(expected, result, expected.equal(result))
