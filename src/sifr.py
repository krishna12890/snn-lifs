import torch


def sifr(x, **kwargs):
    # print("DEBUG: x.shape =", x.shape)
    # ensure there is at least one trailing feature dim
    if x.ndim == 2:  # x has shape (B, T)
        print("DEBUG: x has no trailing dims — adding singleton feature dim")
        x = x.unsqueeze(-1)  # now (B, T, 1)

    if (lam := kwargs.get("lamda", None)) is None:
        raise AssertionError("Missing required keyword argument: 'lamda'")
    if (vth := kwargs.get("vth", None)) is None:
        raise AssertionError("Missing required keyword argument: 'vth'")
    else:
        lam = kwargs.get("lamda", 0.2)
        vth = kwargs.get("vth", 0.5)

    b = 1
    B = x.shape[0]
    T = x.shape[1]
    trailing_shape = x.shape[2:]
    print("trailing dimensions:", trailing_shape)

    Vm = torch.zeros((B, T + 1) + trailing_shape, dtype=x.dtype)
    Vsp_bar = torch.ones((B, T) + trailing_shape, dtype=x.dtype)
    print("before loop: Vm shape:", Vm.shape, "Vsp_bar shape:", Vsp_bar.shape)

    # ...existing code...
    for n in range(1, T):
        prev_vm = Vm[:, n - 1, ...]
        prev_vsp_bar = Vsp_bar[:, n - 1, ...]
        input_x = x[:, n, ...]
        Vm[:, n, ...] = lam * prev_vsp_bar * prev_vm + b * input_x
        Vsp_bar[:, n, ...] = (Vm[:, n, ...] >= vth).float()

        # debug print: values and shapes
        # print(f"n={n}")
        # print(" prev_vm:", prev_vm)
        # print(" prev_vsp_bar:", prev_vsp_bar)
        # print(" input_x:", input_x)
        # print(" Vm[:,n]:", Vm[:, n, ...])
        # print(" Vsp_bar[:,n]:", Vsp_bar[:, n, ...])
        # print("-" * 40)

    return Vm


# if __name__ == "__main__":
#     x = torch.tensor([
#         [
#             [8.0],
#             [-7.0],
#             [4.0],
#             [7.5],
#             ]
#         ])

#     print("x shape:", x.shape)
#     kwargs = {'lamda': 0.2, 'vth': 0.5}
#     output = sifr(x, **kwargs)
#     print("Final Vm:", output)
