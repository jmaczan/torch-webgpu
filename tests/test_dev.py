import torch
import torch_webgpu


if __name__ == "__main__":
    storage = torch.arange(200, device="cpu")
    a = torch.as_strided(storage, size=(2, 3, 4), stride=(20, 6, 1))
    a = a.to("webgpu")
    with torch.no_grad():
        a.reshape((4, 6))
