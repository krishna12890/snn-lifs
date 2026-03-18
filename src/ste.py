import torch


def StraightThroughEstimator(x: torch.Tensor) -> torch.Tensor:
    surrogate = x

    return (
        surrogate
        + (
            torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x)) - surrogate
        ).detach()
    )
