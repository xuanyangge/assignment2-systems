from typing import Any
from builtins import type

import torch
import triton
import math
import triton.language as tl
import einops
from einops import einsum, rearrange
import torch.nn as nn
from functools import partial
import torch.distributed as dist
from cs336_basics.model import Embedding, Linear, RMSNorm
from enum import StrEnum

class LayerType(StrEnum):
    LINEAR = "linear"
    EMBEDDING = "embedding"
    OTHERS = "other"

# Don't forget to go over dtypes.

class FSDPContainer(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.workers = {}
        self.gathered_layer_weights = {}
        self.gradient_buffer = {}

        self.layers = nn.ModuleList([])
        self.non_shard_params = []
        self.gatherable_inds = []
        self.layer_subweights = {}
        self.module = module
        for i, layer in enumerate(module.modules()):
            if isinstance(layer, (Linear, Embedding)):
                self.store_subweight(i, layer)
                layer.register_forward_pre_hook(partial(self._all_gather_prehook, i))
                layer.register_forward_hook(partial(self._forward_hook, i))
                layer.register_full_backward_pre_hook(partial(self._all_gather_prehook, i))
                layer.register_full_backward_hook(partial(self._backward_hook, i))
                layer.weight.register_post_accumulate_grad_hook(partial(self._grad_hook, i))
                self.gatherable_inds.append(i)
            else:
                for p in layer.parameters(recurse=False):
                    self.non_shard_params.append(p)

            self.layers.append(layer)
    
    def store_subweight(self, ind, layer: torch.nn.Module):
        d_out, d_in = layer.weight.shape
        assert(d_out % self.world_size == 0) 
        slice_size = d_out // self.world_size
        layer.weight.data =  layer.weight[slice_size * self.rank: slice_size *(self.rank +1), : ].clone()

        # if self.compute_dtype is not None:
        #     layer.weight.data =  sliced_data.to(self.compute_dtype)
        # else:
        #     layer.weight.data = sliced_data

        self.layer_subweights[ind] = layer.weight.data

    def all_gather_worker(self, idx):
        subweight = self.layer_subweights[idx]
        output = [torch.empty_like(subweight, dtype=self.compute_dtype) for _ in range(self.world_size)]
        self.gathered_layer_weights[idx] = output
        if self.compute_dtype is not None:
            subweight = subweight.to(self.compute_dtype)
            
        return dist.all_gather(self.gathered_layer_weights[idx], subweight, async_op=True)

    def _all_gather_prehook(self, idx, module, args):
        if idx in self.workers:
            self.workers[idx].wait()
            self.workers.pop(idx)
        else:
            self.all_gather_worker(idx).wait()

        module.weight.data = torch.cat(self.gathered_layer_weights[idx], dim = 0)
        self.gathered_layer_weights.pop(idx)

    
    def _forward_hook(self, idx, module, args, output):
        module.weight.data = self.layer_subweights[idx]
        gather_ind = self.gatherable_inds.index(idx)

        gather_ahead = 2
        for i in range(1, gather_ahead + 1):
            if i + gather_ind >= len(self.gatherable_inds):
                break
            
            next_ind = self.gatherable_inds[i + gather_ind]
            self.workers[next_ind] = self.all_gather_worker(next_ind)


    def _backward_hook(self, idx, module, args, output):
        module.weight.data = self.layer_subweights[idx]
        gather_ind = self.gatherable_inds.index(idx)

        gather_back = 2
        for i in range(1, gather_back + 1):
            if gather_ind - i < 0:
                break
            
            next_ind = self.gatherable_inds[gather_ind - i]
            self.workers[next_ind] = self.all_gather_worker(next_ind)

    def _grad_hook(self, idx, tensor):
        output = torch.empty((tensor.grad.shape[0] // self.world_size, tensor.grad.shape[1]), dtype=torch.float32)
        worker = dist.reduce_scatter_tensor(output, tensor.grad.to(torch.float32), op= dist.ReduceOp.AVG, async_op=True)
        self.gradient_buffer[idx] = (worker, output, tensor)


    def resetForwardState(self):
        self.gathered_layer_weights = {}
        self.workers = {}
        self.gradient_buffer = {}

    def forward(self, *inputs, **kwargs):
        self.resetForwardState()
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        [handle.wait() for handle, _, _ in self.gradient_buffer.values()]

        for _, output, tensor in self.gradient_buffer.values():
            tensor.grad = output
        
        gradients = []
        for p in self.non_shard_params:
            if p.grad is not None:
                gradients.append(p.grad)

        #gradients = [p.grad for p in self.module.parameters() if p.grad is not None]
        
        gradients_flattend = torch._utils._flatten_dense_tensors(gradients)
        dist.all_reduce(gradients_flattend, op=dist.ReduceOp.AVG, async_op=False)
        gradients_reduced = torch._utils._unflatten_dense_tensors(gradients_flattend, gradients)

        for i in range(len(gradients)):
            gradients[i].copy_(gradients_reduced[i])


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
