"""
This script is for megatron checkpoint experiment which is not compatible to
current megatron version 0.7.0.
"""

import os
import shutil
import warnings
import functools
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.distributed.checkpoint as dcp

from pathlib import Path
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed._sharded_tensor import ShardedTensor as TorchShardedTensor
from megatron.core.dist_checkpointing import ShardedTensor, save, load
from megatron.core.dist_checkpointing.strategies.torch import _unwrap_pyt_sharded_tensor
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue


warnings.filterwarnings("ignore")


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def print_0(*a, **kw):
    if dist.get_rank() == 0:
        print(*a, **kw)


def print_state_dict(rank, state_dict):
    if dist.get_rank() == rank:
        for k, v in state_dict.items():
            if isinstance(v, TorchShardedTensor):
                print(f"{k}: {v.local_shards()}")
            else:
                print(f"{k}: {v}")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def test_oss_save_load(rank, folder):
    model1 = create_model(rank)
    model2 = create_model(rank)
    with FSDP.state_dict_type(model1, StateDictType.SHARDED_STATE_DICT):
        with FSDP.state_dict_type(model2, StateDictType.SHARDED_STATE_DICT):
            shutil.rmtree(folder, ignore_errors=True)
            Path(folder).mkdir(parents=True, exist_ok=True)

            state_dict1 = model1.state_dict()
            state_dict2 = model2.state_dict()
            print_0("===== before =====")
            print_state_dict(0, state_dict1)
            print_state_dict(0, state_dict2)

            storage_writer = dcp.FileSystemWriter(str(folder))
            storage_reader = dcp.FileSystemReader(str(folder))

            dcp.save(state_dict1, storage_writer=storage_writer)
            dcp.load(state_dict2, storage_reader=storage_reader)

            print_0("===== after =====")
            print_state_dict(0, state_dict1)
            print_state_dict(0, state_dict2)


def test_sync_save_load(rank, folder):
    model1 = create_model(rank)
    model2 = create_model(rank)
    with FSDP.state_dict_type(model1, StateDictType.SHARDED_STATE_DICT):
        with FSDP.state_dict_type(model2, StateDictType.SHARDED_STATE_DICT):
            shutil.rmtree(folder, ignore_errors=True)
            Path(folder).mkdir(parents=True, exist_ok=True)

            state_dict1 = model1.state_dict()
            state_dict2 = model2.state_dict()
            print_0("===== before =====")
            print_state_dict(0, state_dict1)
            print_state_dict(0, state_dict2)

            save(state_dict1, folder, async_sharded_save=False)
            load(state_dict2, folder)

            print_0("===== after =====")
            print_state_dict(0, state_dict1)
            print_state_dict(0, state_dict2)


def test_async_save_load(rank, folder):
    model1 = create_model(rank)
    model2 = create_model(rank)
    with FSDP.state_dict_type(model1, StateDictType.SHARDED_STATE_DICT):
        with FSDP.state_dict_type(model2, StateDictType.SHARDED_STATE_DICT):
            shutil.rmtree(folder, ignore_errors=True)
            Path(folder).mkdir(parents=True, exist_ok=True)

            state_dict1 = model1.state_dict()
            state_dict2 = model2.state_dict()
            print_0("===== before =====")
            print_state_dict(0, state_dict1)
            print_state_dict(0, state_dict2)

            async_calls = AsyncCallsQueue()
            async_request = save(state_dict1, folder, async_sharded_save=True)
            async_calls.schedule_async_request(async_request)
            async_calls.maybe_finalize_async_calls(blocking=True)
            load(state_dict2, folder)

            print_0("===== after =====")
            print_state_dict(0, state_dict1)
            print_state_dict(0, state_dict2)


def create_model(rank):
    features = 32
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy)
    return FSDP(LinearRegression(features, 1).to(rank), auto_wrap_policy=auto_wrap_policy)


def test_save_load(rank, world_size):
    prefix = Path("checkpoints")
    test_oss_save_load(rank, prefix / "oss")
    test_sync_save_load(rank, prefix / "sync")
    test_async_save_load(rank, prefix / "async")


def main(rank, world_size):
    setup(rank, world_size)
    test_save_load(rank, world_size)
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
