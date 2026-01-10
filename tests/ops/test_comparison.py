import torch
import torch_webgpu  # noqa: F401


# Equal tests
def test_eq_tensor():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
    b = torch.tensor([1.0, 5.0, 3.0, 6.0], device="webgpu")
    result = torch.eq(a, b)
    expected = torch.tensor([True, False, True, False])
    assert torch.equal(result.to("cpu"), expected)


def test_eq_scalar():
    a = torch.tensor([1.0, 2.0, 3.0, 2.0], device="webgpu")
    result = torch.eq(a, 2.0)
    expected = torch.tensor([False, True, False, True])
    assert torch.equal(result.to("cpu"), expected)


# Not equal tests
def test_ne_tensor():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
    b = torch.tensor([1.0, 5.0, 3.0, 6.0], device="webgpu")
    result = torch.ne(a, b)
    expected = torch.tensor([False, True, False, True])
    assert torch.equal(result.to("cpu"), expected)


def test_ne_scalar():
    a = torch.tensor([1.0, 2.0, 3.0, 2.0], device="webgpu")
    result = torch.ne(a, 2.0)
    expected = torch.tensor([True, False, True, False])
    assert torch.equal(result.to("cpu"), expected)


# Less than tests
def test_lt_tensor():
    a = torch.tensor([1.0, 5.0, 3.0, 4.0], device="webgpu")
    b = torch.tensor([2.0, 3.0, 3.0, 6.0], device="webgpu")
    result = torch.lt(a, b)
    expected = torch.tensor([True, False, False, True])
    assert torch.equal(result.to("cpu"), expected)


def test_lt_scalar():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
    result = torch.lt(a, 3.0)
    expected = torch.tensor([True, True, False, False])
    assert torch.equal(result.to("cpu"), expected)


# Less than or equal tests
def test_le_tensor():
    a = torch.tensor([1.0, 5.0, 3.0, 4.0], device="webgpu")
    b = torch.tensor([2.0, 3.0, 3.0, 6.0], device="webgpu")
    result = torch.le(a, b)
    expected = torch.tensor([True, False, True, True])
    assert torch.equal(result.to("cpu"), expected)


def test_le_scalar():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
    result = torch.le(a, 3.0)
    expected = torch.tensor([True, True, True, False])
    assert torch.equal(result.to("cpu"), expected)


# Greater than tests
def test_gt_tensor():
    a = torch.tensor([1.0, 5.0, 3.0, 4.0], device="webgpu")
    b = torch.tensor([2.0, 3.0, 3.0, 6.0], device="webgpu")
    result = torch.gt(a, b)
    expected = torch.tensor([False, True, False, False])
    assert torch.equal(result.to("cpu"), expected)


def test_gt_scalar():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
    result = torch.gt(a, 2.0)
    expected = torch.tensor([False, False, True, True])
    assert torch.equal(result.to("cpu"), expected)


# Greater than or equal tests
def test_ge_tensor():
    a = torch.tensor([1.0, 5.0, 3.0, 4.0], device="webgpu")
    b = torch.tensor([2.0, 3.0, 3.0, 6.0], device="webgpu")
    result = torch.ge(a, b)
    expected = torch.tensor([False, True, True, False])
    assert torch.equal(result.to("cpu"), expected)


def test_ge_scalar():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
    result = torch.ge(a, 3.0)
    expected = torch.tensor([False, False, True, True])
    assert torch.equal(result.to("cpu"), expected)
