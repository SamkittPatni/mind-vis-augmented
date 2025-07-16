import math, sys
import torch
import numpy as np
from torch._six import inf
import torch
import tc_lstm.utils as ut
import time

class NativeScalerWithGradNormCount:
    # This class is used to scale the loss and gradients during training, especially for mixed precision training.
    # It helps in preventing underflow and overflow issues by scaling the gradients appropriately.
    state_dict_key = "amp_scaler"

    def __init__(self):
        '''Initialize the scaler for mixed precision training'''
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        '''Scales the loss, computes gradients, clips them if wanted, and updates the optimizer.'''
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        '''Compute the norm of gradients for a list of parameters.'''
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the norm of gradients for a list of parameters.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
    return total_norm

def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                        loss_scaler, include_same_time=False, log_writer=None, config=None, start_time=None):
    """
    Train model one epoch with mixed precision on temporal sym contrast loss.
    """
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    accum_iter = config.accum_iter
    start_time = time.time()

    for data_iter_step, (x, lengths) in enumerate(data_loader):
        # per-iteration LR schedule
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        x = x.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            loss, z = model(x, lengths, include_same_time=include_same_time)  # Forward pass through the model

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping.")
            sys.exit(1)

        # backward + step
        loss_scaler(loss, optimizer, clip_grad=config.clip_grad, parameters=model.parameters())
        optimizer.zero_grad()
        total_loss.append(loss_value)

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 1:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')
    
    return np.mean(total_loss)