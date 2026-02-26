import math

import pytest
import torch

from src.lifs import lifs

BASIC_TEST_PATTERNS = {
    "simple_no_spike": {
        "input": torch.tensor([[0.2, 0.3, 0.4]]),
        "lam": 0.25,
        "vth": 0.5,
        "expected": torch.tensor([[0.0, 0.0, 0.0]]),
    },
    "simple_spike": {
        "input": torch.tensor([[0.6, 0.2, 0.3]]),
        "lam": 0.5,
        "vth": 0.5,
        "expected": torch.tensor([[1.0, 0.0, 0.0]]),
    },
}


@pytest.mark.parametrize("test_pattern", list(BASIC_TEST_PATTERNS.keys()))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lifs_fuctional(test_pattern, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pattern = BASIC_TEST_PATTERNS[test_pattern]
    x = pattern["input"].to(device)
    lam = pattern["lam"]
    vth = pattern["vth"]
    expected = pattern["expected"].to(device)

    spikes = lifs(x, lam, vth)

    assert spikes.shape == expected.shape
    assert torch.allclose(spikes, expected)


def test_lifs_basic_values():
    x = torch.tensor([[3.0, -8.0, 7.0]])  # shape (B=1, T=3, D=0)
    spikes = lifs(x, lam=0.2, vth=0.5)

    # Vm shape should be (B, T+1, *trailing)
    assert spikes.shape == (1, 3)

    # expected values computed by hand:
    # n=0  -> 0.2*1*0 + 3 = 3 -> (1)
    # n=1 -> 0.2*0*3 + (-8)  = -8 -> (0)
    # n=2 -> 0.2*1*7 + 7 = 8.4 -> (1)
    expected = [[1.0, 0.0, 1.0]]  # expected binary spike output at each time-step
    for b in range(spikes.shape[0]):  # Loop B
        for t in range(spikes.shape[1]):  # Loop T
            val = spikes[b, t].item()
            exp = expected[b][t]
            assert math.isclose(
                val, exp, rel_tol=1e-6
            ), f"batch {b}, t={t}: {val} != {exp}"


def test_lifs_trailing_dims_shape():
    # Test data: varied inputs for robust LIF testing
    x_base = torch.tensor([1.2, -0.3, 0.8])
    lam, vth = 0.2, 0.5
    expected_spikes = torch.tensor([1.0, 0.0, 1.0])

    x_base = torch.tensor([1.2, -0.3, 0.8])
    lam, vth = 0.2, 0.5
    expected_spikes = torch.tensor([1.0, 0.0, 1.0])

    # Case 1: (B,T) - auto-unsqueeze
    x = x_base.unsqueeze(0)
    spikes = lifs(x, lam, vth)
    assert spikes.shape == (1, 3)
    assert torch.allclose(spikes, expected_spikes)

    # Case 2: (B,T,1) - single feature
    x = x_base.unsqueeze(0).unsqueeze(-1)
    spikes = lifs(x, lam, vth)
    assert spikes.shape == (1, 3, 1)
    assert torch.allclose(spikes.squeeze(-1), expected_spikes)

    # Case 3: (B,T,2,2) - multi-dimensional features
    x = x_base.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
    spikes = lifs(x, lam, vth)
    assert spikes.shape == (1, 3, 2, 2)
    expected_spikes_reshaped = expected_spikes.view(1, 3, 1, 1).expand(-1, -1, 2, 2)
    assert torch.allclose(spikes, expected_spikes_reshaped)


def test_lifs_requires_kwargs():
    x = torch.zeros(1, 3, 1)
    with pytest.raises(AssertionError):
        lifs(x, lam=None, vth=0.5)
