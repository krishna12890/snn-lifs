import math
import torch
import pytest

from src.sifr import sifr


def test_sifr_basic_values():
    # input shape (1,4,1)
    x = torch.tensor([[3.0, 8.0, 7.0, 4.0]])
    kwargs = {
        "lamda": 0.2,
        "vth": 0.5,
    }

    Vm = sifr(x, **kwargs)

    # Vm shape should be (B, T+1, *trailing)
    assert Vm.shape == (1, 5, 1)

    # expected values computed by hand:
    # n=0 -> 0 (initial)
    # n=1 -> 8
    # n=2 -> 0.2*1*8 + 7 = 8.6
    # n=3 -> 0.2*1*8.6 + 4 = 5.72
    # n=4 -> remains 0 (not updated)
    expected = [0.0, 8.0, 8.6, 5.72, 0.0]

    for i, exp in enumerate(expected):
        val = Vm[0, i, 0].item()
        assert math.isclose(val, exp, rel_tol=1e-6, abs_tol=1e-6), f"index {i}: {val} != {exp}"


def test_sifr_trailing_dims_shape():
    # random input with trailing dims (2, 5, 3, 4)
    B, T, D1, D2 = 2, 5, 3, 4
    x = torch.randn(B, T, D1, D2)

    kwargs = {"lamda": 0.1, "vth": 0.0}

    Vm = sifr(x, **kwargs)

    # Vm should have shape (B, T+1, D1, D2)
    assert Vm.shape == (B, T + 1, D1, D2)

    # basic sanity: first time-step (index 0) is zero
    assert torch.allclose(Vm[:, 0, ...], torch.zeros_like(Vm[:, 0, ...]))


def test_sifr_requires_kwargs():
    x = torch.zeros(1, 3, 1)
    with pytest.raises(AssertionError):
        sifr(x)
