"""Flow-matching loss in the released PyTorch training code."""

import torch
from torch.nn import functional as F

from simplefold.model.flow import LinearPath
from simplefold.utils.boltz_utils import center_random_augmentation, weighted_rigid_align


def clean_prediction(xt, velocity, t):
    return xt + (1 - t[:, None, None]) * velocity


def smooth_lddt(pred, truth, observed, t, cutoff, weight, scheduled):
    true_dist = torch.cdist(truth, truth)
    pairs = (true_dist < cutoff) * (1 - torch.eye(truth.shape[1], device=truth.device))
    pairs = pairs * observed.unsqueeze(1) * observed.unsqueeze(2)
    error = (torch.cdist(pred, pred) - true_dist).abs()
    score = sum(torch.sigmoid(threshold - error) for threshold in (0.5, 1., 2., 4.)) / 4
    per_protein = 1 - (score * pairs).sum((-2, -1)) / pairs.sum((-2, -1)).clamp(min=1)
    if scheduled:
        return (per_protein * (1 + 8 * torch.relu(t - 0.5))).mean()
    return per_protein.mean() * weight


def flow_losses(batch, xt, noise, velocity, t, *, scale, path=None, rigid_align=True,
                lddt_cutoff=15., lddt_weight=1., lddt_scheduled=True):
    path = LinearPath() if path is None else path
    truth = batch['coords']
    observed = batch['atom_resolved_mask'].float()
    pred = clean_prediction(xt, velocity, t)
    if rigid_align:
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            aligned = weighted_rigid_align(truth.detach().float(), pred.detach().float(),
                                          torch.ones_like(observed), observed)
        target = path.compute_ut(t[:, None, None], noise, aligned)
    else:
        target = path.compute_ut(t[:, None, None], noise, truth)
    squared = F.mse_loss(velocity, target, reduction='none') * observed.unsqueeze(-1)
    mse = (squared.sum(1) / observed.sum(1, keepdim=True)).mean()
    total = mse
    geometric_loss = mse.new_zeros(())
    if lddt_weight > 0:
        def to_angstrom(x):
            return center_random_augmentation(x, batch['atom_pad_mask'], augmentation=False) * scale
        geometric_loss = smooth_lddt(to_angstrom(pred), to_angstrom(truth), observed, t,
                                     lddt_cutoff, lddt_weight, lddt_scheduled)
        total = total + geometric_loss
    return {'velocity_target': target, 'pred_clean': pred, 'velocity_mse': mse,
            'smooth_lddt_loss': geometric_loss, 'native_loss': total}
