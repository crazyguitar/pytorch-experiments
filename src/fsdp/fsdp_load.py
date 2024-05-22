import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.checkpoint.state_dict import set_state_dict
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

CHECKPOINT_DIR = "checkpoint"


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_load_example(rank, world_size):
    print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # generates the state dict we will load into
    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict
    }
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=CHECKPOINT_DIR,
    )
    # sets our state dicts on the model and optimizer, now that we've loaded
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state_dict,
        optim_state_dict=optimizer_state_dict
    )

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_load_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
