import os
import time
import torch
import torch.distributed as dist
from torch import multiprocessing  as mp

def worker():
    try:
        os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
        s = time.time()
        rank = dist.get_rank()
        while (time.time() - s) < 30:
            tensor = torch.tensor([rank], device=torch.cuda.current_device())
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            time.sleep(1)
            if dist.get_rank() == 0:
                print(f"subprocess duration: {time.time() - s}")
    finally:
        dist.destroy_process_group()


def main():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        s = time.time()
        while (time.time() - s) < 60:
            tensor = torch.tensor([rank], device=torch.cuda.current_device())
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            time.sleep(1)
            if dist.get_rank() == 0:
                print(f"duration: {time.time() - s}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_nod=8 mp_communication.py
    ctx = mp.get_context("fork")
    p = ctx.Process(target=worker, args=(), daemon=True)
    p.start()
    main()
    p.join()
