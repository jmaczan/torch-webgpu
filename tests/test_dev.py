import torch
import torch_webgpu
from torch_webgpu import webgpu_backend


# @torch.compile(backend=webgpu_backend)
def fn():
    a = torch.tensor([[-1.5, 2.7, 1.0, 2.0], [-1.0, 0.9, 1.1, -2.1]], device="webgpu")
    b = torch.tensor(
        [[-1.0, 0.9], [1.1, -2.1], [-1.5, 2.7], [1.0, 2.0]], device="webgpu"
    )
    print(a.size(), b.size())
    result = torch.mm(a, b)
    # result = torch.relu(result)
    result = result.to("cpu")
    return result


if __name__ == "__main__":
    result = fn()
    expected = torch.tensor([[4.9700, -0.3200], [-1.7600, -4.0200]], device="cpu")
    assert torch.allclose(result, expected)
    print(expected, result, f"allclose: {torch.allclose(result, expected)}")
