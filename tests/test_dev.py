import torch
import torch_webgpu


def main():
    wgpu = torch.empty((4, 4), device="webgpu", dtype=torch.float32)
    wgpu2 = wgpu.to("webgpu")
    cpu = wgpu2.to("cpu")
    print(torch.allclose(wgpu, wgpu2))


if __name__ == "__main__":
    main()
