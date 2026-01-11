import torch
import torch_webgpu  # noqa: F401
import pytest


def test_cpu_to_webgpu():
    """Test moving a CPU tensor to WebGPU using .to()"""
    cpu = torch.tensor([1.0, 2.0, 3.0])
    webgpu = cpu.to('webgpu')
    assert webgpu.device.type == 'privateuseone' or str(webgpu.device).startswith('webgpu')


def test_webgpu_to_cpu():
    """Test moving a WebGPU tensor to CPU using .to()"""
    webgpu = torch.ones(3, device='webgpu')
    cpu = webgpu.to('cpu')
    assert cpu.device.type == 'cpu'
    assert torch.allclose(cpu, torch.ones(3))


def test_round_trip_preserves_values():
    """Test that CPU -> WebGPU -> CPU preserves tensor values"""
    original = torch.tensor([1.5, -2.7, 3.14, 0.0])
    webgpu = original.to('webgpu')
    back_to_cpu = webgpu.to('cpu')
    assert torch.allclose(back_to_cpu, original)


def test_larger_tensor():
    """Test device transfer with larger tensors"""
    cpu = torch.randn(32, 64)
    webgpu = cpu.to('webgpu')
    back = webgpu.to('cpu')
    assert torch.allclose(back, cpu, rtol=1e-5, atol=1e-5)


def test_multidimensional_tensor():
    """Test device transfer with multi-dimensional tensors"""
    cpu = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    webgpu = cpu.to('webgpu')
    back = webgpu.to('cpu')
    assert torch.allclose(back, cpu)


def test_webgpu_to_webgpu_same_device():
    """Test that same-device transfer works (might copy)"""
    webgpu = torch.ones(3, device='webgpu')
    same = webgpu.to('webgpu')
    assert same.device.type == 'privateuseone' or str(same.device).startswith('webgpu')
    # Values should be preserved
    assert torch.allclose(same.to('cpu'), webgpu.to('cpu'))


def test_cpu_method():
    """Test the .cpu() method"""
    webgpu = torch.ones(3, device='webgpu')
    cpu = webgpu.cpu()
    assert cpu.device.type == 'cpu'
    assert torch.allclose(cpu, torch.ones(3))


def test_webgpu_method():
    """Test the .webgpu() method (if available)"""
    cpu = torch.tensor([1.0, 2.0, 3.0])
    if hasattr(cpu, 'webgpu'):
        webgpu = cpu.webgpu()
        assert webgpu.device.type == 'privateuseone' or str(webgpu.device).startswith('webgpu')
    else:
        pytest.skip(".webgpu() method not available")


@pytest.mark.skip(reason="Empty tensors not yet supported in WebGPU backend")
def test_empty_tensor():
    """Test device transfer with empty tensors"""
    cpu = torch.empty(0)
    webgpu = cpu.to('webgpu')
    back = webgpu.to('cpu')
    assert back.shape == torch.Size([0])


def test_scalar_tensor():
    """Test device transfer with scalar tensors"""
    cpu = torch.tensor(3.14)
    webgpu = cpu.to('webgpu')
    back = webgpu.to('cpu')
    assert torch.allclose(back, cpu)
