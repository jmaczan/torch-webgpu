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
    g = torch.nn.GELU()
    b = g(b)
    b = b.to("cpu")
    print(b, b.device)
    # b = b.to("webgpu")
    # c = b + b
    # print(c.device)
    # c = c.to("cpu")
    # print(c)
