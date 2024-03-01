#%%
import os
from typing import Callable

import torch
import torch.distributed as dist


def init_process(rank: int,
                 size: int,
                 fn: Callable[[int, int], None],
                 backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


import torch.multiprocessing as mp


def do_broadcast(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    src = 3
    if rank == src:
        tensor = torch.tensor([rank], dtype=torch.float32)
    else:
        tensor = torch.empty(1)
        # sending all tensors to the others
    dist.broadcast(tensor, src=src, group=group)
    # all ranks will have tensor([0.]) from rank 0
    print(f"[{rank}] data = {tensor}")


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_broadcast))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
