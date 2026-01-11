import torch

__all__ = ["is_initialized", "current_device", "get_amp_supported_dtype"]


def is_initialized():
    return True


def current_device():
    return 0


def get_amp_supported_dtype():
    """Return list of supported dtypes for AMP on WebGPU.

    WebGPU primarily uses float32, so we only support float32 for now.
    """
    return [torch.float32]
