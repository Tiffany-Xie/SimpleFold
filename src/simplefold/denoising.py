"""One-step probes and standard PyTorch sampling on the same linear path."""

import torch

from simplefold.model.flow import LinearPath
from simplefold.model.objective import flow_losses
from simplefold.model.torch.sampler import EMSampler
from simplefold.processor.protein_processor import COORD_SCALE
from simplefold.utils.boltz_utils import center_random_augmentation


def noise_draw(batch, seed):
    x = batch['coords']
    generator = torch.Generator(device=x.device).manual_seed(seed)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


@torch.no_grad()
def one_step(model, batch, t, noise, *, stage='finetune'):
    """t=0 is noise and t=1 is data. Velocities are in model units per unit t."""
    truth = batch['coords']
    t = torch.as_tensor(t, device=truth.device, dtype=truth.dtype).expand(truth.shape[0])
    if not torch.all((t >= 0) & (t <= 1)) or stage not in ('pretrain', 'finetune'):
        raise ValueError('Expected t in [0,1] and a published training stage')
    if noise.shape != truth.shape or noise.dtype != truth.dtype or noise.device != truth.device:
        raise ValueError('noise must match the padded model coordinates')
    _, xt, _ = LinearPath().interpolant(t, noise, truth)
    velocity = model(xt, t, batch)['predict_velocity']
    out = flow_losses(batch, xt, noise, velocity, t, scale=COORD_SCALE,
                      lddt_scheduled=stage == 'finetune')
    pred = out.pop('pred_clean')
    return {**out, 'loss_stage': stage, 't': t, 'noise': noise, 'velocity': velocity, 'xt_angstrom': xt * COORD_SCALE,
            'truth_angstrom': truth * COORD_SCALE, 'pred_clean_angstrom': pred * COORD_SCALE}


@torch.no_grad()
def sample(model, batch, *, seed=42, steps=500, tau=0.1, trajectory=False, snapshot_times=None):
    """Official log-time SDE, optionally recording full trajectories or nearest-time snapshots."""
    if steps < 1 or tau < 0:
        raise ValueError('Expected positive steps and nonnegative tau')
    coords = batch['coords']
    generator = torch.Generator(device=coords.device).manual_seed(seed)
    noise = torch.randn(coords.shape, dtype=coords.dtype, device=coords.device, generator=generator)
    sampler = EMSampler(num_timesteps=steps, t_start=1e-4, tau=tau, log_timesteps=True, w_cutoff=0.99)
    selected = []
    if snapshot_times is not None:
        if trajectory or not len(snapshot_times) or any(not 0 <= t < 1 for t in snapshot_times):
            raise ValueError('Specify snapshot times in [0,1) or request a full trajectory')
        selected = sorted({int((sampler.steps[:-1] - t).abs().argmin()) for t in snapshot_times})
    states, inputs, predictions, velocities = [], [], [], []
    if trajectory:
        states.append((COORD_SCALE * noise).cpu())

    step = 0
    state = noise

    def record(t, t_next, model_input, velocity, pred, next_state):
        nonlocal step, state
        if trajectory or step in selected:
            inputs.append((model_input * COORD_SCALE).cpu())
            predictions.append((pred * COORD_SCALE).cpu())
            states.append(((next_state if trajectory else state) * COORD_SCALE).cpu())
            if trajectory:
                velocities.append(velocity.cpu())
        step += 1
        state = next_state

    out = sampler.sample(model, LinearPath(), noise, batch, generator=generator,
                         record_step=record if trajectory or selected else None, progress=False)
    final = center_random_augmentation(out['denoised_coords'], batch['atom_pad_mask'], augmentation=False)
    result = {'final_angstrom': final * COORD_SCALE, 'times': sampler.steps, 'tau': tau, 'seed': seed}
    if trajectory:
        result.update(states_angstrom=torch.stack(states), inputs_angstrom=torch.stack(inputs),
                      predictions_angstrom=torch.stack(predictions), velocities=torch.stack(velocities))
    elif selected:
        result.update(snapshot_indices=torch.tensor(selected), snapshot_times=sampler.steps[selected],
                      states_angstrom=torch.stack(states), inputs_angstrom=torch.stack(inputs),
                      predictions_angstrom=torch.stack(predictions))
    return result
