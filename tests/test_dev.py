import torch
import torch_webgpu
from torch_webgpu import webgpu_backend


def fn():
    a = torch.tensor([-1.5, 2.7, 1.0, 2.0], device="webgpu")
    b = torch.tensor([-1.0, 0.9, 1.1, -2.1], device="webgpu")
    result = torch.mul(a, b)
    result = torch.relu(result)
    result = result.to("cpu")
    return result


if __name__ == "__main__":
    result = fn()
    expected = torch.tensor([1.5, 2.43, 1.1, 0], device="cpu")
    assert torch.allclose(result, expected)
    print(expected, result, f"allclose: {expected.equal(result)}")
