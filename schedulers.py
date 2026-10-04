import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math

class CustomOneCycleLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy='cos',
                 div_factor=25, final_div_factor=1e4, momentum_range=(0.85, 0.95), annihilation_frac=0.1, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.momentum_range = momentum_range
        self.annihilation_steps = int(total_steps * annihilation_frac)
        
        # Compute phase lengths
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up - self.annihilation_steps

        # Compute initial and final learning rates
        self.initial_lrs = [max_lr / div_factor for _ in optimizer.param_groups]
        self.final_lrs = [max_lr / final_div_factor for _ in optimizer.param_groups]

        # Set initial learning rates
        for param_group, initial_lr in zip(optimizer.param_groups, self.initial_lrs):
            param_group['lr'] = initial_lr

        # Store initial momentum values
        self.initial_momentums = [momentum_range[1] for _ in optimizer.param_groups]

        # Set optimizer momentum to max value
        for param_group in optimizer.param_groups:
            if 'betas' in param_group:  # Adam-like optimizers
                param_group['betas'] = (momentum_range[1], param_group['betas'][1])
            elif 'momentum' in param_group:  # SGD-like optimizers
                param_group['momentum'] = momentum_range[1]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.step_size_up:
            # Warm-up phase (increase LR)
            scale = step / self.step_size_up
        elif step < self.step_size_up + self.step_size_down:
            # Cool-down phase (decrease LR)
            scale = 1 - (step - self.step_size_up) / self.step_size_down
        elif step >= self.total_steps - 2:
            # Last two epochs have the same learning rate as the final epoch
            return self.final_lrs
        else:
            # Annihilation phase (drastically reduce LR)
            annihilation_scale = (self.total_steps - step) / self.annihilation_steps
            return [final_lr + annihilation_scale * (base_lr - final_lr) for base_lr, final_lr in zip(self.initial_lrs, self.final_lrs)]

        if self.anneal_strategy == 'cos':
            lr_factor = 0.5 * (1 + math.cos(math.pi * (1 - scale)))  # Cosine annealing
        else:
            lr_factor = scale  # Linear decay

        return [base_lr + lr_factor * (self.max_lr - base_lr) for base_lr in self.initial_lrs]

    def get_momentum(self):
        step = self.last_epoch

        if step < self.step_size_up:
            # Warm-up phase (decrease momentum)
            scale = 1 - step / self.step_size_up
        elif step < self.step_size_up + self.step_size_down:
            # Cool-down phase (increase momentum)
            scale = (step - self.step_size_up) / self.step_size_down
        elif step >= self.total_steps - 2:
            # Last two epochs: keep momentum at final value
            return [self.momentum_range[0] for _ in self.optimizer.param_groups]
        else:
            # Annihilation phase (momentum remains at min)
            scale = 1

        return [self.momentum_range[0] + scale * (self.momentum_range[1] - self.momentum_range[0])
                for _ in self.optimizer.param_groups]

    def step(self):
        super().step()

        # Update momentum
        momentums = self.get_momentum()
        for param_group, momentum in zip(self.optimizer.param_groups, momentums):
            if 'betas' in param_group:  # Adam-like optimizers
                param_group['betas'] = (momentum, param_group['betas'][1])
            elif 'momentum' in param_group:  # SGD-like optimizers
                param_group['momentum'] = momentum
                
def get_scheduler(optimizer, ds_name = '', name="ExponentialLR", step_size=1000, epochs=25):
    """
    Returns a learning rate scheduler with fixed parameters based on the given schedule type.

    Args:
        optimizer: PyTorch optimizer.
        ds_name: Name of dataset.
        name (str): Name of the scheduler. Options: 
            'ExponentialLR', 'StepLR', 'MultiStepLR', 'CosineAnnealingLR', 
            'OneCycleLR', 'CyclicLR', 'ReduceLROnPlateau'.
        step_size (int): Step size for applicable schedulers.

    Returns:
        A PyTorch learning rate scheduler.
    """
    
    if name == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  # 0.8, 0.85, 0.9, 0.95, 0.99

    elif name == "StepLR":  # bad
        # epochs // 3
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    elif name == "MultiStepLR":  # bad
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)

    elif name == "CosineAnnealingLR":
        # eta_min: Minimum learning rate. Default: 0.
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    elif name == "OneCycleLR":
        # (self, optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy='cos', 
        # div_factor=25, final_div_factor=1e4, momentum_range=(0.85, 0.95), annihilation_frac=0.1, last_epoch=-1):
                     
        return CustomOneCycleLR(
            optimizer, 
            max_lr=1e-3,            # Fixed max_lr
            total_steps=step_size,  # Total steps for annealing
            anneal_strategy='cos',
            annihilation_frac=0.1
        )
        
        '''return optim.lr_scheduler.OneCycleLR(
            # epochs, steps_per_epoch 
            optimizer, 
            max_lr=3e-3,  # Fixed max_lr
            total_steps=step_size,  # Total steps for annealing
            anneal_strategy='cos',
            div_factor = 3,
            #final_div_factor=1e4 
        )'''
    
    elif name == "CyclicLR": 
        
        return optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=3e-6,  # Lower bound
            max_lr=3e-3,  # Upper bound
            step_size_up=step_size,  # Steps to peak LR
            mode='triangular2'
        )
        
    else:
        raise ValueError(f"Invalid schedule type '{name}'. Choose from: ExponentialLR, StepLR, MultiStepLR, CosineAnnealingLR, OneCycleLR, CyclicLR.")
