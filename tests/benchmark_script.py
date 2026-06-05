import torch
import argparse
from pathlib import Path
from cs336_basics.model import BasicsTransformerLM  # or whatever you need
import numpy.typing as npt
import numpy as np
import torch
from cs336_basics.nn_utils import cross_entropy
import timeit
from cs336_basics.optimizer import AdamW
import torch.cuda.nvtx as nvtx

def parse_args():
    p = argparse.ArgumentParser(description="LLM training loop")

    # Model architecture
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--context_length", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=48)
    p.add_argument("--d_model", type=int, default=1600)
    p.add_argument("--num_heads", type=int, default=25)
    p.add_argument("--d_ff", type=int, default=4288)

    # Optimizer (AdamW)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)

    # Training loop
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=10_000)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--checkpoint_interval", type=int, default=1_000)
    p.add_argument("--log_interval", type=int, default=10)

    # # Paths
    # p.add_argument("--train_data", type=Path, required=True)
    # p.add_argument("--eval_data", type=Path, required=True)
    # p.add_argument("--checkpoint_dir", type=Path, required=True)
    # p.add_argument("--resume_from", type=Path, default=None)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    # 0 forward only, 1 forward and backward, 2 forward, backward and optimize.
    p.add_argument("--mode", type=int, default = 0)

    p.add_argument("--rope_theta", type=float, default=10000)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
    ).to(args.device)
    return model

def generate_random_batch(args):
    seq = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length + 1)).to(args.device)
    return (seq[:, :-1], seq[:,1:])

def run_test_operation(args, model, random_batch, optimizer):
    if args.mode == 0:
        model(random_batch[0])
    elif args.mode == 1:
        logits = model(random_batch[0])
        loss = cross_entropy(logits, random_batch[1])
        loss.backward()
        model.zero_grad()
    elif args.mode == 2:
        logits = model(random_batch[0])
        loss = cross_entropy(logits, random_batch[1])
        loss.backward()
        optimizer.step()
        model.zero_grad()

    model.zero_grad()





def main():
    args = parse_args()
    set_seed(args.seed)
    random_batch = generate_random_batch(args)
    model = build_model(args)
    optimizer = AdamW(model.parameters())

    print("Model Building Done")
    # warmup 
    for i in range(args.warmup):
        run_test_operation(args, model, random_batch, optimizer)

    torch.cuda.synchronize()  
    times = []
    for i in range(args.iters):
        with nvtx.range("step"):
            torch.cuda.synchronize()
            start = timeit.default_timer()  
            run_test_operation(args, model, random_batch, optimizer)
            torch.cuda.synchronize()
            end = timeit.default_timer()    
            elapsed = end - start
            times.append(elapsed)
    
    print(f"average {np.mean(times)*1000:.2f} ms/step")
    print(f"std {np.std(times, ddof=1)*1000:.2f} ms/step")

if __name__ == "__main__":
    main()
