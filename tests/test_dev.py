import torch
import torch_webgpu


def main():
    cpu = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    wgpu = cpu.to("webgpu")

    print("cpu:\n", cpu)
    print("adding two tensors on webgpu!")
    x = wgpu + wgpu
    print(wgpu.device)
    print(wgpu.size())
    print(wgpu.shape)
    print(x.cpu())


if __name__ == "__main__":
    main()
