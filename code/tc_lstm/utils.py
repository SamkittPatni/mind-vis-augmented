import torch
import torch.nn.functional as F
import os
import math

def symmetric_info_nce_loss(z: torch.Tensor, temperature=0.07) -> torch.Tensor:
    """
    Compute symmetric InfoNCE loss over adjacent time embeddings.
    z: (B, T, D)
    """
    B, T, D = z.shape
    N = B * T
    # Flatten
    z_flat = z.view(N, D)  # (N, D)

    # Cosine similarity matrix
    sim = F.cosine_similarity(
        z_flat.unsqueeze(1), z_flat.unsqueeze(0), dim=-1
    ) / temperature  # (N, N)

    # Exponential
    exp_sim = torch.exp(sim)
    # Build time index for each flat pos
    time_idx = torch.arange(T, device=z.device).unsqueeze(0).repeat(B, 1).view(-1)
    # Mask self similarities
    diag_mask = torch.eye(N, device=z.device).bool()
    # Mask same-time negatives
    same_time = time_idx.unsqueeze(1) == time_idx.unsqueeze(0)  # (N,N)

    # Identify positive pairs: forward and backward adjacent
    anchors = []
    positives = []
    for b in range(B):
        base = b * T
        for t in range(T - 1):
            anchors.append(base + t)
            positives.append(base + t + 1)
            anchors.append(base + t + 1)
            positives.append(base + t)

    anchors = torch.tensor(anchors, device=z.device)
    positives = torch.tensor(positives, device=z.device)
    numer = exp_sim[anchors, positives] # Numerators: exp_sim[anchors, positives]
    # Build negative mask per anchor
    neg_mask = ~(diag_mask | same_time) # valid_neg = ~diag_mask & ~same_time
    neg_mask[anchors, positives] = False # exclude the actual positive from the negatives

    denom = exp_sim[anchors.unsqueeze(1), torch.where(neg_mask[anchors])[1]].view(numer.size(0), -1).sum(dim=1)
    loss = -torch.log(numer / denom) # Loss per positive pair
    return loss.mean()

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