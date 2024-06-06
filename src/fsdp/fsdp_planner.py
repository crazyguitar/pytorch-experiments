import os
import warnings
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed._shard.sharded_tensor as sharded_tensor

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.checkpoint.default_planner import create_default_local_save_plan
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.api import _shard_tensor
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)

warnings.filterwarnings("ignore")


def print_0(*a, **kw):
    if dist.get_rank() == 0:
        print(*a, **kw)


def to_sharded(tensor, num_shard):
    spec = ChunkShardingSpec(
        dim=0, placements=[f"rank:{r}/cuda:{r}" for r in range(num_shard)]
    )
    return _shard_tensor(tensor, spec)


def to_tensor(rank, dst, sharded, h, w):
    tensor = torch.zeros(h, w).to(rank) if rank == dst else None
    sharded.gather(dst, tensor)
    return tensor


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


def test_sharded_tensor(rank, world_size):
    h = 8
    w = 8
    tensor = torch.rand(h, w).to(rank)
    sharded = to_sharded(tensor, world_size)
    ensumble_tensor = to_tensor(rank, 0, sharded, h, w)
    print_0(f"original tensor: {tensor}")
    print_0(f"ensumble_tensor: {ensumble_tensor}")
    print(f"[rank:{rank}] local_shards {sharded.local_shards()}")


def test_sharded_tensor_rand(rank, world_size):
    h = 8
    w = 1
    spec = ChunkShardingSpec(
        dim=0, placements=[f"rank:{r}/cuda:{r}" for r in range(world_size)]
    )
    sharded = sharded_tensor.rand(spec, (h, w))
    print(f"[rank:{rank}] {sharded.local_shards()} size: {sharded.size()}")
    tensor = to_tensor(rank, 0, sharded, h, w)
    print_0(tensor)


def test_local_plan(rank, world_size):
    features = 32
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy)
    model = FSDP(
        LinearRegression(features, 1).to(rank),
        auto_wrap_policy=auto_wrap_policy,
    )
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = model.state_dict()
        plan = create_default_local_save_plan(state_dict, False)
        for item in plan.items:
            print_0(f"index: {item.index}")
            print_0(f"tensor_data.chunk: {item.tensor_data.chunk}")
            print_0(f"tensor_data.properties: {item.tensor_data.properties}")


def main(rank, world_size):
    setup(rank, world_size)
    test_sharded_tensor(rank, world_size)
    test_sharded_tensor_rand(rank, world_size)
    test_local_plan(rank, world_size)
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
