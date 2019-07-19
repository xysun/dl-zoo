import torch
import torch.optim

def reset_lr_scheduler(scheduler: torch.optim.lr_scheduler.MultiStepLR):
    '''
    reset the LR scheduler used in retraining, see Algorithm 1
    '''
    scheduler.base_lrs = list(map(lambda group: group['initial_lr'], scheduler.optimizer.param_groups))
    last_epoch = 0
    scheduler.last_epoch = last_epoch
    scheduler.step(last_epoch)