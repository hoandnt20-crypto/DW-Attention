import os
import torch
import torch.distributed as dist

def setup_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt