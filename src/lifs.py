import torch


def lifs(x, lam, vth):
    # ensure there is at least one trailing feature dim
    original_unsqueezed = False
    if x.ndim == 2:  # x has shanpe (B, T)
        x = x.unsqueeze(-1)  # now (B, T, 1)
        original_unsqueezed = True

    if lam is None:
        raise AssertionError("Missing required keyword argument: 'lamda'")
    if vth is None:
        raise AssertionError("Missing required keyword argument: 'vth'")

    B = x.shape[0]
    T = x.shape[1]
    trailing_shape = x.shape[2:]

    Vm = torch.zeros((B, T) + trailing_shape, dtype=x.dtype)

    for n in range(T):
        prev_vm = (
            torch.zeros((B, T) + trailing_shape, dtype=x.dtype)
            if n == 0
            else Vm[:, n - 1, ...]
        )
        input_x = x[:, n, ...]
        Vm[:, n, ...] = (
            input_x if ((prev_vm < lam).all()) else (lam * prev_vm + input_x)
        )

    output = torch.where(Vm >= vth, 1.0, 0.0)  # binary spike output

    if original_unsqueezed:
        output = output.squeeze(-1)  # remove the singleton feature dim if we added it

    return output
