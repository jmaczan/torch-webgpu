import torch
import torch_webgpu


def main():
    cpu = torch.empty((4, 4), dtype=torch.float32)
    wgpu = cpu.to("webgpu")
    back = cpu.to("cpu")
    print(torch.allclose(cpu, back))


if __name__ == "__main__":
    main()
