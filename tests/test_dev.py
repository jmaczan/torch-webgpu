import torch
import torch_webgpu


if __name__ == "__main__":
    a = torch.tensor(
        [
            [-2.3, 5.2, -999],
            [-5.6, -3333.2, 321.6],
        ],
    )
    print(a)
    b = a.to("webgpu")
    print(b.device)
    b = b.relu()
    b = b.to("cpu")
    print(b, b.device)
