import os
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.checkpoint.default_planner import create_default_local_save_plan


def print_0(*a, **kw):
    if dist.get_rank() == 0:
        print(*a, **kw)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def test_local_plan(model):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = model.state_dict()
        plan = create_default_local_save_plan(state_dict, False)
        for item in plan.items:
            print_0(f"index: {item.index}")
            print_0(f"tensor_data.chunk: {item.tensor_data.chunk}")
            print_0(f"tensor_data.properties: {item.tensor_data.properties}")


def main(rank, world_size):
    setup(rank, world_size)
    features = 32
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy)
    model = FSDP(
        LinearRegression(features, 1).to(rank),
        auto_wrap_policy=auto_wrap_policy,
    )

    test_local_plan(model)
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running test save plan on {world_size} devices.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
