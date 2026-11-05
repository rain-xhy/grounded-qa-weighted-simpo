"""Externally gap-weighted SimPO with matched, window-normalized gradients."""
import math
from .simpo import simpo_loss, backward_pair


def validate_weights(weights):
    if not weights or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in weights):
        raise ValueError('Pair weights must be nonempty, finite, and strictly positive')
    return float(sum(weights))


def weighted_loss(chosen_nll, rejected_nll, weights, beta=10., gamma_beta_ratio=.3):
    import torch
    values = list(weights)
    denominator = validate_weights(values)
    losses = simpo_loss(chosen_nll, rejected_nll, beta, gamma_beta_ratio, 0.)
    if losses.ndim != 1 or losses.numel() != len(values):
        raise ValueError('Exactly one scalar loss is required per preference pair')
    weights = torch.tensor(values, device=losses.device, dtype=losses.dtype)
    return (weights * losses).sum() / denominator


def backward_weighted_pair(loss_fn, weight, window_weight_sum, beta=10., gamma_beta_ratio=.3, scaler=None):
    """Caller freezes weights over the ENTIRE gradient accumulation window."""
    validate_weights([weight, window_weight_sum])
    if weight > window_weight_sum:
        raise ValueError('A pair weight exceeds its accumulation window total')
    result = backward_pair(loss_fn, beta=beta, gamma_beta_ratio=gamma_beta_ratio,
                           sft_weight=0., accumulation=window_weight_sum/weight, scaler=scaler)
    return {**result, 'weighted_loss_contribution': result['loss']*weight/window_weight_sum,
            'sample_weight': weight, 'window_weight_sum': window_weight_sum}
