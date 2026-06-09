from __future__ import annotations

import torch
import triton
import math
import triton.language as tl
import einops
import torch.nn as nn
import torch.distributed as dist

def custom_cdiv(a, b):
    if a % b == 0:
        return a // b
    return a // b  + 1

class flashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False):
        tile_size = 16
        batch_shape = Q.shape[:-2]
        Q = einops.rearrange(Q, "... seq_len d -> (...) seq_len d")
        K = einops.rearrange(K, "... seq_len d -> (...) seq_len d")
        V = einops.rearrange(V, "... seq_len d -> (...) seq_len d")

        batch_size = Q.shape[0]
        
        N_q = Q.shape[-2]
        N_k = V.shape[-2]
        d = Q.shape[-1]
        t_q = custom_cdiv(N_q , tile_size)
        t_k = custom_cdiv(N_k, tile_size)

        O_with_batch = torch.zeros((batch_size, N_q, d))
        L_with_batch = torch.zeros((batch_size, N_q,))

        for b in range(batch_size):
            inner_q = Q[b,:,:]
            inner_k = K[b,:,:]
            inner_v = V[b,:,:]
            
            tile_size = 16
            # Your implementation should take input 𝑸, 𝑲, and 𝑽 as well as a flag is_causal and produce 
            # the output 𝑶 and the logsumexp value 𝐿. You can ignore the is_causal flag for this task. 
            # The autograd.Function forward should then save 𝐿,𝑄,𝐾,𝑉,𝑂 for the backward pass and 
            # return 𝑂

            O = torch.zeros((N_q, d))
            L = torch.zeros((N_q, ))
            # i and j has to be > 1
            for i in range(t_q):
                Q_i = inner_q[i * tile_size: (i+1) * tile_size:, :]
                prev_o = torch.zeros((tile_size,d))
                prev_l = torch.zeros((tile_size,))
                prev_m = torch.full((tile_size,), float('-inf'))
                
                for j in range(t_k):
                    K_j = inner_k[j * tile_size: (j+1) * tile_size, :]
                    V_j = inner_v[j* tile_size: (j+1) * tile_size, :]
                    S_j = Q_i @ K_j.T / math.sqrt(d)

                    row_max = torch.max(S_j, dim = 1).values

                    if j == 0:
                        cur_m = row_max
                    else:
                        cur_m = torch.max(torch.stack((row_max, prev_m)), dim =0).values
                    
                    # S_j is 16 * 16, cur_m is 16*1 
                    P_j = torch.exp(S_j - cur_m[:, None])
                    
                    cur_l = torch.zeros((tile_size,))
                    cur_o = torch.zeros((tile_size,d)) 
                    if j != 0:
                        max_diff_exp = torch.exp(prev_m - cur_m)
                        cur_l = max_diff_exp * prev_l
                        cur_o = torch.diag(max_diff_exp) @ prev_o
                    
                    cur_l += torch.sum(P_j, 1)
                    cur_o += P_j @ V_j

                    prev_o = cur_o
                    prev_l = cur_l
                    prev_m = cur_m

                O[i* tile_size: (i+1) * tile_size, :] = torch.diag(1 /cur_l) @ cur_o
                L[i* tile_size: (i+1) * tile_size] = cur_m + torch.log(cur_l)
            
            O_with_batch[b] = O
            L_with_batch[b] = L
        
        O_with_batch = O_with_batch.reshape(*batch_shape, N_q, d)
        L_with_batch = L_with_batch.reshape(*batch_shape, N_q)
        Q = Q.reshape(*batch_shape, N_q, d)
        K = K.reshape(*batch_shape, N_k, d)
        V = V.reshape(*batch_shape, N_k, d)
        ctx.save_for_backward(Q, K, V, O_with_batch, L_with_batch)
        return O_with_batch


class ddp(nn.Module):
    def __init__(self, module:torch.nn.Module):
        super().__init__()
        self.module = module
        return 
    
    def forward(self, data):
        return self.module(data)
    
    def backward(self):
        self.module.backward()
    
    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
        
class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False):
        D = Q.shape[-1]
        k_tile_size = q_tile_size = 16

        batch_size = Q.shape[0]
        N_q = Q.shape[-2]
        N_k = V.shape[-2]
        d = Q.shape[-1]

        O = torch.zeros((batch_size, N_q, d), device="cuda")
        L = torch.zeros((batch_size, N_q,), device="cuda")

 
        flash_fwd_kernel[(triton.cdiv(N_q, q_tile_size), Q.shape[0])] (
            Q, K, V, 
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            math.sqrt(D),
            D,
            q_tile_size, k_tile_size,
            is_causal= is_casual
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_casual = is_casual
        return O
    
    def backward(ctx):
        raise NotImplementedError

# To debug, we suggest comparing the results of each Triton operation you perform with the 
# tiled PyTorch implementation you wrote in part (a).
# • Your launch grid should be set as (𝑇𝑞,batch_size), meaning each Triton program instance 
# will load only elements from a single batch index, and only read/write to a single query 
# tile of 𝑸, 𝑶, and 𝐿.
# • The kernel should only have a single loop, which will iterate key tiles 1≤𝑗≤𝑇𝑘.
# • Advance block pointers at the end of the loop

# where scale is  1√𝑑 and Q_TILE_SIZE and K_TILE_SIZE are 𝐵𝑞 and 𝐵𝑘 respectively. You can 
# tune these later.
# These additional guidelines may help you avoid precision issues:
# • The on chip buffers (𝑶𝑖,𝑙,𝑚) should have dtype tl.float32. If you’re accumulating into an 
# output buffer, use the acc argument (acc = tl.dot(..., acc=acc)).
# • Cast  ̃𝑷(𝑗)
# 𝑖  to the dtype of 𝑽(𝑗) before multiplying them, and cast 𝑶𝑖 to the appropriate 
# dtype before writing it to global memory. Casting is done with tensor.to. You can get the 
# dtype of a tensor with tensor.dtype, and the dtype of a block pointer/pointer with 
# *_block_ptr.type.element_ty


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)
    )

    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    prev_o = tl.zeros((Q_TILE_SIZE, D) , dtype= tl.float32)
    prev_l = tl.zeros((Q_TILE_SIZE,) , dtype= tl.float32)
    prev_m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype = tl.float32)
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    for j in range(tl.cdiv(N_QUERIES, K_TILE_SIZE)):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s_j = tl.dot(q_i, tl.trans(k_j)) / scale

        if is_causal:
            q_inds = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_inds = j * K_TILE_SIZE  + tl.arange(0, K_TILE_SIZE)
            mask = q_inds[:, None] >= k_inds[None, :]
            # i span from quer
            s_j = tl.where(mask, s_j, s_j-1e6)


        row_max = tl.max(s_j, axis = 1)

        if j == 0:
            cur_m = row_max
        else:
            cur_m = tl.maximum(row_max, prev_m)
        
        p_j = tl.exp(s_j - cur_m[:, None])
        p_j = p_j.to(v_j.type.element_ty)

        cur_o = tl.zeros((Q_TILE_SIZE, D) , dtype= tl.float32)
        cur_l = tl.zeros((Q_TILE_SIZE,) , dtype= tl.float32)

        if j!= 0:
            max_diff_exp = tl.exp(prev_m - cur_m)
            cur_l = max_diff_exp * prev_l
            #  I can't use diag either
            cur_o =  prev_o * max_diff_exp[:, None] 

        
        cur_l += tl.sum(p_j, 1)
        cur_o += tl.dot(p_j, v_j)

        prev_o = cur_o
        prev_l = cur_l
        prev_m = cur_m

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    

    tl.store(O_block_ptr, prev_o * ((1 / prev_l)[:, None]), boundary_check=(0, 1))
    tl.store(L_block_ptr, prev_m + tl.log(prev_l))



def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    return flashAttention


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    return  MyTritonFlashAttentionAutogradFunctionClass


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDP(module)
    for p in module.parameters():
        dist.broadcast(p.data, src=0)
        
    return ddp(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    fully-sharded data parallel training, including weight sharding,
    all-gather for forward/backward, and gradient reduce-scatter.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with FSDP.
        compute_dtype: optional torch.dtype
            If provided, weights are cast to this dtype before communication
            and compute, saving bandwidth. Master weights stay in fp32.
    Returns:
        Instance of an FSDP class.
    """
    # For example: return FSDP(module, compute_dtype=compute_dtype)
    raise NotImplementedError


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the FSDP-wrapped model.
    """
    # For example: fsdp_model.finish_gradient_synchronization()
    raise NotImplementedError


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors. Replicated parameters are returned as-is.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
    Returns:
        State dictionary mapping parameter names to full (unsharded) tensors.
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
