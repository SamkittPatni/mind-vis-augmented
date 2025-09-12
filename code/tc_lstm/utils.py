import torch
import torch.nn.functional as F
import os
import math

import torch
import torch.nn.functional as F

def symmetric_info_nce_loss(
    z: torch.Tensor,
    temperature: float = 0.07,
    include_same_time: bool = True
) -> torch.Tensor:
    """
    Memory-efficient symmetric InfoNCE over adjacent time embeddings.
    z: (B, T, D)
    """
    B, T, D = z.shape

    # normalize once
    z_norm = F.normalize(z, p=2, dim=2)       # (B, T, D)
    z_flat = z_norm.view(B * T, D)            # (B*T, D)

    device = z.device
    batch_offsets = torch.arange(B, device=device) * T

    total_loss = 0.0
    total_pairs = 0

    # for each adjacent pair (forward and backward)
    for t in range(T - 1):
        for i, j in ((t, t+1), (t+1, t)):
            # anchors = z_norm[:, i] ? (B, D)
            sim = torch.matmul(z_norm[:, i], z_flat.t()) / temperature   # (B, B*T)
            exp_sim = sim.exp()                                          # (B, B*T)

            # sum over all positions
            sum_all = exp_sim.sum(dim=1)                                 # (B,)

            # indices into the flattened sequence
            self_idx = batch_offsets + i                                 # (B,)
            pos_idx  = batch_offsets + j                                 # (B,)

            pos_exp = exp_sim[torch.arange(B, device=device), pos_idx]   # (B,)

            if include_same_time:
                # exclude *all* embeddings at this same time i
                # time_idx == self_idx is exactly the B positions at time=i
                same_time_sum = exp_sim[:, self_idx].sum(dim=1)          # (B,)
                denom = sum_all - same_time_sum - pos_exp
            else:
                # exclude only self
                self_exp = exp_sim[torch.arange(B, device=device), self_idx]
                denom = sum_all - self_exp - pos_exp

            loss = -torch.log(pos_exp / denom)  # (B,)
            total_loss += loss.sum()
            total_pairs += B

    return total_loss / total_pairs


def adjust_learning_rate(optimizer, epoch, config):
    """
    This function adjusts the learning rate of the optimizer based on the current epoch.
    The learning rate is scaled linearly during the warmup phase and then follows a cosine decay schedule for better and stable convergence.
    """
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs # Slowly bring lr from 0 to config.lr
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs))) # Moves lr down to min_lr
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def save_model(config, epoch, model, optimizer, loss_scaler, checkpoint_paths):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, 'checkpoint.pth'))
    

def load_model(config, model, checkpoint_path ):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded with {checkpoint_path}')