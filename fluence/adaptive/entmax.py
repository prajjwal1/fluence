# EntmaxBisectFunction has been adapted from https://github.com/deep-spin/entmax/
__all__ = ["AlphaChooser", "EntmaxAlpha", "EntmaxBisectFunction", "entmax_bisect"]

import torch
from torch import nn
from torch.autograd import Function


class AlphaChooser(torch.nn.Module):
    """
    It manages the alpha values in alpha-entmax
    function.
    """

    def __init__(self, head_count):
        super(AlphaChooser, self).__init__()
        self.pre_alpha = nn.Parameter(torch.randn(head_count))

    def forward(self):
        alpha = 1 + torch.sigmoid(self.pre_alpha)
        return torch.clamp(alpha, min=1.01, max=2)


class EntmaxAlpha(nn.Module):
    def __init__(self, head_count, dim=0):
        super(EntmaxAlpha, self).__init__()
        self.dim = dim
        self.alpha_chooser = nn.Parameter(AlphaChooser(head_count)())
        self.alpha = self.alpha_chooser

    def forward(self, att_scores):
        batch_size, head_count, query_len, key_len = att_scores.size()

        expanded_alpha = self.alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # [1,nb_heads,1,1]
        expanded_alpha = expanded_alpha.expand((batch_size, -1, query_len, 1))
        # [bs, nb_heads, query_len,1]
        p_star = entmax_bisect(att_scores, expanded_alpha)
        return p_star


class EntmaxBisectFunction(Function):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        X = X * (alpha - 1)

        max_val, _ = X.max(dim=dim, keepdim=True)

        tau_lo = max_val - cls._gp(1, alpha)
        tau_hi = max_val - cls._gp(1 / d, alpha)

        f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)

        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:

            # alpha gradient computation
            # d_alpha = (partial_y / partial_alpha) * dY
            # NOTE: ensure alpha is not close to 1
            # since there is an indetermination
            # batch_size, _ = dY.shape

            # shannon terms
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            # shannon entropy
            ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)

            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)

        return dX, d_alpha, None, None, None


def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    """
    alpha-entmax: normalizing sparse transform (a la softmax).
    Solves the optimization problem:
        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.
    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.
    This function is differentiable with respect to both X and alpha.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        alpha.shape[j] == (X.shape[j] if j != dim else 1)
        A value of alpha=2 corresponds to sparsemax, and alpha=1 corresponds to
        softmax (but computing it this way is likely unstable).
    dim : int
        The dimension along which to apply alpha-entmax.
    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.
    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)
