"""
test on megatron version 0.7.0.
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
from typing import Optional, Tuple, Union

from pathlib import Path
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed._sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)

from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncCallsQueue,
    AsyncRequest,
)

from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistSaveShardedStrategy,
    TorchDistLoadShardedStrategy,
)

from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
)

from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
    save_state_dict_async_plan,
)

from megatron.core.dist_checkpointing.dict_utils import (
    extract_matching_values,
    dict_list_map_inplace,
)

from megatron.core.dist_checkpointing.utils import (
    extract_nonpersistent,
)

from megatron.core.dist_checkpointing.core import (
    CheckpointingConfig,
    maybe_load_config,
    save_config,
)

from megatron.core.dist_checkpointing.strategies.base import (
    AsyncSaveShardedStrategy,
    LoadCommonStrategy,
    LoadShardedStrategy,
    SaveCommonStrategy,
    SaveShardedStrategy,
)

from megatron.core.dist_checkpointing.mapping import (
    CheckpointingException,
    ShardedStateDict,
    StateDict,
)


warnings.filterwarnings("ignore")


class PyTorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
    def async_save(
        self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
    ) -> AsyncRequest:
        writer = FileSystemWriterAsync(checkpoint_dir, thread_count=self.thread_count)
        save_state_dict_ret = save_state_dict_async_plan(
            sharded_state_dict,
            writer,
            None,
            planner=DefaultSavePlanner(),
        )
        return self._get_save_and_finalize_callbacks(writer, save_state_dict_ret)


class PyTorchDistLoadShardedStrategy(TorchDistLoadShardedStrategy):
    def load(
        self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
    ) -> StateDict:
        dcp.load_state_dict(
            sharded_state_dict,
            dcp.FileSystemReader(checkpoint_dir),
            planner=DefaultLoadPlanner(),
        )
        return sharded_state_dict


def save(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[SaveShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[SaveCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
    async_sharded_save: bool = False,
) -> Optional[AsyncRequest]:

    checkpoint_dir = Path(checkpoint_dir)
    sharded_strategy = PyTorchDistSaveShardedStrategy(backend="torch_dist", version=1)
    _, sharded_state_dict = extract_nonpersistent(sharded_state_dict)

    def metadata_finalize_fn():
        if torch.distributed.get_rank() == 0:
            save_config(
                CheckpointingConfig(sharded_strategy.backend, sharded_strategy.version),
                checkpoint_dir,
            )
        torch.distributed.barrier()

    if not async_sharded_save:
        sharded_strategy.save(sharded_state_dict, checkpoint_dir)
        metadata_finalize_fn()
        return

    if not isinstance(sharded_strategy, AsyncSaveShardedStrategy):
        raise CheckpointingException(
            f"Cannot apply async_save to non-async strategy {sharded_strategy}"
        )
    async_request = sharded_strategy.async_save(sharded_state_dict, checkpoint_dir)
    async_request.finalize_fns.append(metadata_finalize_fn)
    return async_request


def load(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
) -> StateDict:
    sharded_strategy = PyTorchDistLoadShardedStrategy()
    checkpoint_dir = Path(checkpoint_dir)
    return sharded_strategy.load(sharded_state_dict, checkpoint_dir)


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
    FSDP.state_dict_type(model1, StateDictType.SHARDED_STATE_DICT)
    FSDP.state_dict_type(model2, StateDictType.SHARDED_STATE_DICT)

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
    FSDP.state_dict_type(model1, StateDictType.SHARDED_STATE_DICT)
    FSDP.state_dict_type(model2, StateDictType.SHARDED_STATE_DICT)

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
    FSDP.state_dict_type(model1, StateDictType.SHARDED_STATE_DICT)
    FSDP.state_dict_type(model2, StateDictType.SHARDED_STATE_DICT)

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


def test_dcp_save_megatron_load(rank, folder):
    model1 = create_model(rank)
    model2 = create_model(rank)
    FSDP.set_state_dict_type(model1, StateDictType.SHARDED_STATE_DICT)
    FSDP.set_state_dict_type(model2, StateDictType.SHARDED_STATE_DICT)

    Path(folder).mkdir(parents=True, exist_ok=True)
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    print_0("===== before =====")
    print_state_dict(0, state_dict1)
    print_state_dict(0, state_dict2)

    storage_writer = dcp.FileSystemWriter(str(folder))
    dcp.save(state_dict1, storage_writer=storage_writer)
    load(state_dict2, folder)

    print_0("===== after =====")
    print_state_dict(0, state_dict1)
    print_state_dict(0, state_dict2)


def test_megatron_save_dcp_load(rank, folder):
    model1 = create_model(rank)
    model2 = create_model(rank)
    FSDP.set_state_dict_type(model1, StateDictType.SHARDED_STATE_DICT)
    FSDP.set_state_dict_type(model2, StateDictType.SHARDED_STATE_DICT)

    Path(folder).mkdir(parents=True, exist_ok=True)
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    print_0("===== before =====")
    print_state_dict(0, state_dict1)
    print_state_dict(0, state_dict2)

    save(state_dict1, folder, async_sharded_save=False)
    storage_reader = dcp.FileSystemReader(str(folder))
    dcp.load(state_dict2, storage_reader=storage_reader)

    print_0("===== after =====")
    print_state_dict(0, state_dict1)
    print_state_dict(0, state_dict2)


def create_model(rank):
    features = 32
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy)
    return FSDP(
        LinearRegression(features, 1).to(rank), auto_wrap_policy=auto_wrap_policy
    )


def test_save_load(rank, world_size):
    prefix = Path("checkpoints", ignore_errors=True)
    if dist.get_rank() == 0:
        shutil.rmtree(prefix, ignore_errors=True)
        prefix.mkdir(parents=True, exist_ok=True)

    torch.distributed.barrier()
    test_oss_save_load(rank, prefix / "1")
    test_sync_save_load(rank, prefix / "2")
    test_async_save_load(rank, prefix / "3")
    test_dcp_save_megatron_load(rank, prefix / "4")
    test_megatron_save_dcp_load(rank, prefix / "5")


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
