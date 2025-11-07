import torch
import torch_webgpu


def main():
    cpu_src = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    # CPU -> WebGPU
    wgpu = cpu_src.to("webgpu")

    # Optional: WebGPU -> WebGPU
    wgpu2 = wgpu.to("webgpu")

    # WebGPU -> CPU
    cpu = wgpu.to("cpu")
    cpu2 = wgpu2.to("cpu")

    print("cpu:\n", cpu)
    print("cpu2:\n", cpu2)
    print("allclose:", torch.allclose(cpu, cpu2))
    print("vs original:", torch.allclose(cpu, cpu_src), torch.allclose(cpu2, cpu_src))
    x = wgpu + wgpu
    print(x)


if __name__ == "__main__":
    main()
