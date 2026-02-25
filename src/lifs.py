import torch


def lifs(x, lam, vth):
    if lam is None:
        raise AssertionError("Missing required keyword argument: 'lamda'")
    if vth is None:
        raise AssertionError("Missing required keyword argument: 'vth'")

    B = x.shape[0]
    T = x.shape[1]
    trailing_shape = x.shape[2:]

    # initialization
    Vm = torch.zeros((B, T) + trailing_shape, dtype=x.dtype)

    for n in range(T):
        prev_vm = (
            torch.zeros((B, 1) + trailing_shape, dtype=x.dtype)
            if n == 0
            else Vm[:, n - 1, ...]
        )
        input_x = x[:, n, ...]
        Vm[:, n, ...] = torch.where(prev_vm < vth, lam * prev_vm + input_x, input_x)

        # debug print: values and shapes
        print(f"n={n}")
        print(" prev_vm:", prev_vm)
        print(" input_x:", input_x)
        print(" Vm[:,n]:", Vm[:, n, ...])
        print("-" * 40)

    output = torch.where(Vm >= vth, 1.0, 0.0)  # binary spike output

    return output
