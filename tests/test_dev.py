import torch
import torch_webgpu


def main():
    print(torch._C._get_privateuse1_backend_name())

    dev = torch.device("webgpu")
    print(dev)

    dev = torch.device("webgpu:0")
    print(dev)

    empty = torch.empty((4, 4), device="webgpu")
    print(empty.device, empty.shape)
    empty_cpu = empty.to("cpu")
    print(empty_cpu.device, empty_cpu.shape)
    # print(torch._C._dispatch_dump_table("aten::view"))

    # print(empty)


if __name__ == "__main__":
    main()
