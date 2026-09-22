import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed():
    """Reads the torchrun environment variables. Returns (rank, local_rank, world_size)."""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend=os.environ.get('DIST_BACKEND', 'nccl'))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def setup_logging(log_file, rank):
    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)
    return logging.getLogger('macos')


def reduce_mean(value, world_size, device):
    if world_size <= 1:
        return value
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


def clean_state_dict(state_dict):
    """Strips the DataParallel 'module.' prefix and drops sampling-grid buffers saved by older versions of the code."""
    return {k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in state_dict.items() if not k.endswith('spatial_transformer.grid')}


def unwrap(model):
    return model.module if hasattr(model, 'module') else model
