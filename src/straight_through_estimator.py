import torch


class SpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, vth, slope: float = 10.0):
        """
        STE for binary spiking: hard 0/1 forward, sigmoid surrogate backward.
        Handles scalar vth automatically.
        """
        # Handle scalar vth → tensor on same device/dtype
        if not torch.is_tensor(vth):
            vth = torch.tensor(vth, device=u.device, dtype=u.dtype)

        # Hard binary spike (passes your assertion)
        out = (u >= vth).to(u.dtype)

        # Save for backward
        ctx.save_for_backward(u, vth)
        ctx.slope = slope
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """grad_out: incoming gradient from downstream"""
        u, vth = ctx.saved_tensors
        slope = ctx.slope

        # Surrogate: derivative of sigmoid(slope*(u-vth))
        s = torch.sigmoid(slope * (u - vth))
        grad_u = grad_out * slope * s * (1.0 - s)

        return grad_u, None, None
