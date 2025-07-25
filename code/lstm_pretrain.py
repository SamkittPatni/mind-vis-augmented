import os
import sys
import time
import datetime
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
import torch.nn.functional as F
import timm.optim.optim_factory as optim_factory
import wandb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy

from config import Config_LSTM_fMRI
from dataset import LSTM_HCP_dataset, lstm_collate_fn
from tc_lstm.lstm_for_fmri import LSTMforFMRI
from tc_lstm.trainer import NativeScalerWithGradNormCount as NativeScaler
from tc_lstm.trainer import train_one_epoch
from tc_lstm.utils import save_model, adjust_learning_rate  # load_model if needed

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

class wandb_logger:
    def __init__(self, config):
        wandb.init(
            project="mind-vis-augmented",
            anonymous="allow",
            group='tc-lstm',
            config=config,
            reinit=True
        )

        self.config = config
        self.step = None

    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step

    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)


def get_args_parser():
    parser = argparse.ArgumentParser('LSTM pre-training for fMRI', add_help=False)
    # Training parameters
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--num_epoch', type=int,   help='number of pretrain epochs')
    parser.add_argument('--batch_size', type=int,   help='batch size')

    parser.add_argument('--input_dim', type=int, default=1, help='input dimension of the LSTM')
    parser.add_argument('--hidden_size', type=int, default=1024, help='hidden size of the LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional LSTM')

    # Project settings
    parser.add_argument('--root_path', type=str,  help='project root path')
    parser.add_argument('--seed', type=int,  help='random seed')
    parser.add_argument('--roi', type=str,  help='ROI name for dataset')
    parser.add_argument('--aug_times', type=int,  help='augmentation times')
    parser.add_argument('--num_sub_limit', type=int,  help='max subjects to include')
    parser.add_argument('--include_hcp', type=bool, help='include HCP data')
    parser.add_argument('--include_kam', type=bool, help='include KAM data')
    parser.add_argument('--resume_from', type=str, default=None, help='path to resume from checkpoint')

    # Distributed training parameters
    parser.add_argument('--local_rank', type=int, default=0, help='local process rank')
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def main(config):
    # Set up distributed training if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')

    # Output path for saving results
    output_path = os.path.join(config.root_path, 'results', 'lstm_pretrain',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)

    # Set random seeds for reproducibility
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Dataset preparation
    dataset_pretrain = LSTM_HCP_dataset(path=os.path.join(config.root_path, 'data/HCP/npz'), roi=config.roi, 
                                        normalize=True, window_size=config.window_size, window_stride=config.window_stride, 
                                        num_sub_limit=config.num_sub_limit)
    print(f'Dataset size: {len(dataset_pretrain)}\nNumber of voxels: {dataset_pretrain.num_voxels}')
    sampler = DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None

    dataloader = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, 
                            shuffle=(sampler is None), collate_fn=lstm_collate_fn, pin_memory=True)
    
    # Model initialization
    config.num_voxels = dataset_pretrain.num_voxels
    model = LSTMforFMRI(input_dim=dataset_pretrain.num_voxels, hidden_size=config.hidden_size, 
                        num_layers=config.num_layers, bidirectional=config.bidirectional)
    model.to(device)
    model_without_ddp = model
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)
    
    # Optimizer and learning rate scheduler
    param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Resume from checkpoint if specified
    start_epoch = 0
    if config.resume_from is not None and os.path.isfile(config.resume_from):
        print(f"Resuming from checkpoint: {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device)
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        loss_scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1

    # Logger setup
    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    # Training loop
    start_time = time.time()
    print("Start LSTM pre-training ...")

    for epoch in range(start_epoch, config.num_epoch):
        if torch.cuda.device_count() > 1:
            sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, loss_scaler, log_writer=logger, 
                                     config=config, start_time=start_time, include_same_time=False)
        
        # log per-epoch metrics
        if config.local_rank == 0:
            logger.log('train_loss_epoch', train_loss, step=epoch)

        # checkpoint every 20 epochs and at the end
        if (epoch + 1) % 20 == 0 or (epoch + 1) == config.num_epoch:
            if config.local_rank == 0:
                ckpt_dir = os.path.join(output_path, 'checkpoints')
                save_model(config, epoch, model_without_ddp, optimizer, loss_scaler, ckpt_dir)
                plot_lstm_diagnostics(model_without_ddp, dataloader, device, epoch + 1, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    if logger is not None:
        logger.finish()

@torch.no_grad()
def plot_lstm_diagnostics(model, dataloader, device, epoch, logger):
    """
    Plot three diagnostic figures for a single batch:
      1) Histogram of embedding norms
      2) Cosine-similarity heatmap of one sequence
      3) 2D PCA trajectory of embeddings over time
    """
    model.eval()
    # grab one batch
    batch = next(iter(dataloader))
    x = batch['fmri'].to(device)       # (B, T, V)
    lengths = batch['length']
    # forward: we ignore the loss, only need embeddings
    _, z = model(x, lengths)
    # pick the first sequence: (T, D)
    z0 = z[0].detach().cpu()           # torch.Tensor

    # 1) Embedding norm histogram
    norms = z0.norm(dim=1).numpy()     # shape (T,)
    fig, ax = plt.subplots()
    ax.hist(norms, bins=50)
    ax.set_title(f'Epoch {epoch}: Embedding Norms')
    logger.log('lstm/embedding_norms', wandb.Image(fig), step=epoch)
    plt.close(fig)

    # 2) Cosine-sim heatmap
    S = F.cosine_similarity(z0.unsqueeze(1), z0.unsqueeze(0), dim=-1).numpy()  # (T,T)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(S, vmin=-1, vmax=1, cmap='RdBu_r')
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Epoch {epoch}: Temporal Cosine Similarity')
    logger.log('lstm/sim_heatmap', wandb.Image(fig), step=epoch)
    plt.close(fig)

    # 3) 2D PCA trajectory
    Z = z0.numpy()                     # (T, D)
    proj = PCA(2).fit_transform(Z)     # (T, 2)
    fig, ax = plt.subplots()
    ax.plot(proj[:,0], proj[:,1], '-o', markersize=3)
    ax.set_title(f'Epoch {epoch}: Embedding Trajectory (PCA)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    logger.log('lstm/trajectory', wandb.Image(fig), step=epoch)
    plt.close(fig)

    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LSTM Pretrain', parents=[get_args_parser()])
    args = parser.parse_args()
    config = Config_LSTM_fMRI()
    # override config attributes with CLI args
    for k, v in vars(args).items():
        if v is not None and hasattr(config, k):
            setattr(config, k, v)
    main(config)