import torch


def ste_lifs(x: torch.Tensor) -> torch.Tensor:
    surrogate = x

    return (
        surrogate
        + (
            torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x)) - surrogate
        ).detach()
    )
