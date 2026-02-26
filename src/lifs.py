import torch


def lifs(x, lam, vth):
    if lam is None:
        raise AssertionError("Missing required keyword argument: 'lambda'")
    if vth is None:
        raise AssertionError("Missing required keyword argument: 'vth'")

    # B = x.shape[0]
    T = x.shape[1]
    # trailing_shape = x.shape[2:]

    # initialization
    Vm = torch.zeros_like(x)  # shape (B, T, *trailing)

    for n in range(T):
        prev_vm = torch.zeros_like(x[:, 0]) if n == 0 else Vm[:, n - 1, ...]
        input_x = x[:, n, ...]
        Vm[:, n, ...] = torch.where(prev_vm < vth, lam * prev_vm + input_x, input_x)

    output = torch.where(Vm >= vth, 1.0, 0.0)  # binary spike output

    return output
