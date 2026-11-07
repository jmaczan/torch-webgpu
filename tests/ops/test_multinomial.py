"""torch.multinomial on webgpu: distribution, replacement semantics,
determinism and (asynchronously reported) invalid inputs."""

import pytest
import torch
import torch_webgpu  # noqa: F401

D = "webgpu"
P = torch.tensor([0.1, 0.0, 0.3, 0.6])


@pytest.fixture(autouse=True)
def no_fallback():
    before = set(torch_webgpu._C._fallback_ops())
    yield
    assert not set(torch_webgpu._C._fallback_ops()) - before


def freq(samples, n=4):
    return torch.bincount(samples.reshape(-1), minlength=n).float() / samples.numel()


def test_with_replacement_frequencies():
    torch.manual_seed(0)
    s = torch.multinomial(P.to(D), 200000, replacement=True).cpu()
    assert s.shape == (200000,) and s.dtype == torch.int64
    torch.testing.assert_close(freq(s), P, atol=5e-3, rtol=0)


def test_single_sample_rows():
    torch.manual_seed(1)
    s = torch.multinomial(P.expand(100000, 4).contiguous().to(D), 1).cpu()
    assert s.shape == (100000, 1)
    torch.testing.assert_close(freq(s), P, atol=5e-3, rtol=0)


def test_without_replacement():
    torch.manual_seed(2)
    s = torch.multinomial(P.expand(20000, 4).contiguous().to(D), 3).cpu()
    assert all(len(set(r.tolist())) == 3 for r in s[:1000])
    assert not (s == 1).any()  # zero probability is never drawn while others remain
    torch.testing.assert_close(freq(s[:, 0]), P, atol=1e-2, rtol=0)


def test_deterministic_with_seed_and_generator():
    p = torch.rand(10, 50, device=D)
    torch.manual_seed(5)
    a = torch.multinomial(p, 4).cpu()
    torch.manual_seed(5)
    b = torch.multinomial(p, 4).cpu()
    assert torch.equal(a, b)
    g1 = torch.Generator(device=D).manual_seed(3)
    g2 = torch.Generator(device=D).manual_seed(3)
    assert torch.equal(torch.multinomial(p, 2, True, generator=g1).cpu(), torch.multinomial(p, 2, True, generator=g2).cpu())


@pytest.mark.parametrize("bad", [[0.0, 0.0], [0.5, -0.1], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_invalid_distribution_raises(bad):
    with pytest.raises(RuntimeError, match="invalid multinomial distribution"):
        torch.multinomial(torch.tensor(bad, device=D), 1).cpu()
    assert torch.ones(3, device=D).sum().item() == 3  # device still usable


def test_argument_checks():
    with pytest.raises(RuntimeError):
        torch.multinomial(torch.tensor([0.5, 0.5], device=D), 3)  # too many without replacement
    with pytest.raises(RuntimeError, match="invalid multinomial"):
        torch.multinomial(torch.tensor([0.5, 0.0, 0.5], device=D), 3).cpu()
