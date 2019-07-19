import torch
from torch.optim.optimizer import Optimizer

import numpy as np

import math
from functools import partial
import pdb

class INQScheduler(object):
    def __init__(self, optimizer: Optimizer, iterative_steps, weight_bits:int):
        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.weight_bits = weight_bits

        # initialize binary quantization mask
        for group in self.optimizer.param_groups:
            group['masks'] = []
            for p in group['params']: # tensor of weight
                if p.requires_grad is False:
                    group['masks'].append(0)
                    continue
                group['masks'].append(torch.ones_like(p.data))

        # remember the index to iterative_steps
        self.idx = 0

        self.calculate_ns()

    def calculate_ns(self):
        '''
        calculate n1, n2 for each tensor
        '''
        for group in self.optimizer.param_groups:
            group['ns'] = [] # a list of (n1,n2) tuples, each tuple for a tensor
            for p in group['params']: # p is a tensor
                if not p.requires_grad:
                    group['ns'].append((0,0))
                    continue
                # formulae (3), (2)
                s = torch.max(torch.abs(p.data)).item() # `item` converts one element tensor to Python scalars
                n_1 = math.floor(math.log((4*s)/3, 2))
                n_2 = int(n_1 + 1 - (2**(self.weight_bits-1))/2)
                group['ns'].append((n_1, n_2))
    
    def quantize(self):
        '''
        quantize the weights according to group['masks']
        '''
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue
                mask = group['masks'][idx]
                ns = group['ns'][idx]
                device = p.data.device
                quantizer = partial(self.f_quantize, n_1 = ns[0], n_2 = ns[1])
                fully_quantized = p.data.clone().cpu().apply_(quantizer).to(device)
                # we quantize & freeze the larger values, see Figure 2
                # todo: duplicate computation; we have already quantized 50% of them
                p.data = torch.where(mask == 0, fully_quantized, p.data)
    
    def f_quantize(self, weight, n_1, n_2):
        '''
        function to be applied on tensor, see formula (4)
        weight is a tensor
        TODO: test
        '''
        alpha = 0
        beta = 2 ** n_2
        abs_weight = math.fabs(weight)
        quantized_weight = 0

        for i in range(n_2, n_1 + 1):
            if (abs_weight >= (alpha + beta)/2) and abs_weight < (3*beta/2):
                quantized_weight = math.copysign(beta, weight)
            alpha = 2 ** i
            beta = 2 ** (i+1)
        return quantized_weight

    def step(self):
        '''
        update group['masks']
        update weights to quantized
        '''
        # update group['masks']
        print("quantization scheduler step:", self.iterative_steps, self.idx)
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']): # p is weight of each component
                prev_mask = group['masks'][idx]
                if p.requires_grad is False:
                    continue
                zeros = torch.zeros_like(p.data)
                ones = torch.ones_like(p.data)
                # calculate threshold
                # todo: this can potentially overwrite previous group['masks']
                quantile = np.quantile(torch.abs(p.data.cpu()).numpy(), 1 - self.iterative_steps[self.idx])
                # we set the mask of weights >= threshold to 0, because they will be quantized and frozen
                new_mask = torch.where(torch.abs(p.data) >= quantile, zeros, ones)
                # compare old and new masks: has any 0 in old turned to be 1?

                group['masks'][idx] = new_mask
                # pdb.set_trace()

        # quantize weights
        self.quantize()
        # pdb.set_trace()
        # increment idx
        self.idx += 1