from typing import Any
from builtins import type

import torch
import triton
import math
import triton.language as tl
import einops
import torch.nn as nn
import torch.distributed as dist
from cs336_basics.model import Embedding, Linear, RMSNorm


class FSDPContainer(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.layers = []
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.current_gather_layer = 0
        self.gathered_layer_weights = {}
        self.workers = {}

        for layer in module.modules():
            if isinstance(layer, Linear):
                d_out, d_in = layer.weight.shape
                assert(d_out % self.world_size == 0) 
                slice_size = d_out // self.world_size
                sliced_layer = Linear(d_in, slice_size)
                sliced_layer.weight = layer.weight[slice_size * self.rank + reminder: slice_size * (self.rank + 1) + reminder, :].to(compute_dtype)
                
                self.layers.append(sliced_layer)
            elif isinstance(layer, Embedding):
                vocab, d_model = layer.weight.shape
                reminder = vocab % self.world_size
                

                # we could refactor this but meh.
                if self.rank < reminder:
                    slice_size  = vocab // self.world_size + 1
                    sliced_layer = Embedding(slice_size, d_model)
                    sliced_layer.weight = layer.weight[slice_size * self.rank: slice_size * (self.rank + 1), :].to(compute_dtype)

                else: 
                    slice_size = d_in // self.world_size
                    sliced_layer = Embedding(slice_size, d_model)
                    sliced_layer.weight = layer.weight[slice_size * self.rank + reminder: slice_size * (self.rank + 1) + reminder, :].to(compute_dtype)
                
                self.layers.append(sliced_layer)
            else:
                self.layers.append(layer)

    def resetForwardState(self):
        self.current_gather_layer = 0
        self.gathered_layer_weights = {}
        self.workers = {}

    def enqueueAllGather(self):
        while self.current_gather_layer < len(self.layers):
            i = self.current_gather_layer
            if isinstance(self.layers[i], (Linear, Embedding)): 
                self.gathered_layer_weights[i] = []
                worker = dist.all_gather(self.gathered_layer_weights[i], self.layers[i].weight, async_op=True)
                self.workers[i] = worker
                break
            else:
                self.current_gather_layer +=1


    # How would backward work in this case? If the intermediate created layers are destoryed? 

    def forward(self, *inputs, **kwargs):
        self.resetForwardState()

        look_ahead = 2
        for _ in range(look_ahead):
            self.enqueueAllGather()
        res = inputs

        for i, layer in enumerate(self.layers):
            self.enqueueAllGather()
            
            self.workers[i].wait()
            updated_layer = layer
            if isinstance(self.layers[i], Linear):
                gradients_flattend = torch.cat(self.gathered_layer_weights[i], dim=0)
                updated_layer = Linear(gradients_flattend.shape[1], gradients_flattend.shape[0])
                updated_layer.weight = gradients_flattend
            elif isinstance(self.layers[i], Embedding):
                gradients_flattend = torch.cat(self.gathered_layer_weights[i], dim=0)
                updated_layer = Embedding(gradients_flattend.shape[0], gradients_flattend.shape[1])
                updated_layer.weight = gradients_flattend
            
            if i in self.gathered_layer_weights:
                self.gathered_layer_weights.clear(i)
                self.workers.clear(i)

            res = updated_layer(*res, **kwargs)

        return res 
    
    def finish_gradient_synchronization(self):
        for hook in self._hooks:
            hook()


# Problem (fsdp):  Fully-Sharded Data Parallel (15 points)
# 38
# Implement a Python class for fully-sharded data parallel training. The class should wrap an
# arbitrary PyTorch nn.Module (your full model) and hook into or wrap any Linear or Embedding
# layer within it. We recommend the following public interface:
# def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
# Given an instantiated PyTorch nn.Module to be parallelized, construct an FSDP module that
# will handle weight all-gathers and gradient reduce-scatters. Make sure that your hooks or your
# module wrappers all-gather the weights in time for the forward pass. To limit memory use,
# only start gathering after the layer two before the current one has completed its forward pass.
# In the backward pass, your hooks or module wrappers should all-gather to have the weights
# available for the computation. When the gradients are available, they should be reduce-
# scattered to the appropriate ranks. Make sure to free the gathered weights after use. When
# compute_dtype is provided, cast the weights to that dtype before communicating or using them
# for compute, while keeping master weights and optimizer updates in FP32.
# def forward(self, *inputs, **kwargs): Calls the wrapped module’s forward() method with the
# provided positional and keyword arguments.
# def finish_gradient_synchronization(self): When called, wait for asynchronous communication
# calls to finish on the GPU.
# Deliverable: Implement a container class to handle fully sharded data parallel training. Each
# shard of this container should be compatible with the standard AdamW implementation from
# assignment 1. To test your FSDP implementation, implement the adapter [adapters.get_fsdp] .
# Run the tests with uv run pytest tests/test_fsdp.py. We recommend running the tests multiple
# times (e.g., 5) to catch any race conditions
