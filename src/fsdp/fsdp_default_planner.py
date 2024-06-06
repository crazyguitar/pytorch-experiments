import os
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed._shard.api import _shard_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.checkpoint.default_planner import (
    _create_default_local_metadata,
    create_default_local_save_plan,
    create_default_local_load_plan
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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def test_save_planner(state_dict):
    plan = create_default_local_save_plan(state_dict, False)
    for write_item in plan.items:
        print_0(write_item)
        # WriteItem(
        #    index=MetadataIndex(
        #       fqn='sharded',
        #       offset=torch.Size([0, 0]), index=None),
        #       type=<WriteItemType.SHARD: 2>,
        #       tensor_data=TensorWriteData(
        #           chunk=ChunkStorageMetadata(
        #               offsets=torch.Size([0, 0]),
        #               sizes=torch.Size([1, 1])
        #           ),
        #           properties=TensorProperties(dtype=torch.float32,
        #           layout=torch.strided,
        #           requires_grad=False,
        #           memory_format=torch.contiguous_format,
        #           pin_memory=False
        #       ),
        #       size=torch.Size([8, 1])
        #    )
        # )
        data = find_state_dict_object(state_dict, write_item.index)
        print_0(data)
        # tensor([[0.2028]], device='cuda:0')


def test_load_planner(state_dict):
    metadata = _create_default_local_metadata(state_dict)
    print_0(metadata)
    # Metadata(
    #   state_dict_metadata={
    #       'sharded': TensorStorageMetadata(
    #           properties=TensorProperties(
    #               dtype=torch.float32,
    #               layout=torch.strided,
    #               requires_grad=False,
    #               memory_format=torch.contiguous_format, pin_memory=False
    #           ),
    #           size=torch.Size([8, 1]),
    #           chunks=[
    #               ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([1, 1])),
    #               ChunkStorageMetadata(offsets=torch.Size([1, 0]), sizes=torch.Size([1, 1])),
    #               ChunkStorageMetadata(offsets=torch.Size([2, 0]), sizes=torch.Size([1, 1])),
    #               ChunkStorageMetadata(offsets=torch.Size([3, 0]), sizes=torch.Size([1, 1])),
    #               ChunkStorageMetadata(offsets=torch.Size([4, 0]), sizes=torch.Size([1, 1])),
    #               ChunkStorageMetadata(offsets=torch.Size([5, 0]), sizes=torch.Size([1, 1])),
    #               ChunkStorageMetadata(offsets=torch.Size([6, 0]), sizes=torch.Size([1, 1])),
    #               ChunkStorageMetadata(offsets=torch.Size([7, 0]), sizes=torch.Size([1, 1]))
    #           ]
    #       )
    #   },
    #   planner_data=None,
    #   storage_data=None
    # )
    plan = create_default_local_load_plan(state_dict, metadata)
    for read_item in plan.items:
        print_0(read_item)
        # ReadItem(
        #   type=<LoadItemType.TENSOR: 1>,
        #   dest_index=MetadataIndex(fqn='sharded', offset=torch.Size([0, 0]), index=0),
        #   dest_offsets=torch.Size([0, 0]),
        #   storage_index=MetadataIndex(fqn='sharded', offset=torch.Size([0, 0]), index=0),
        #   storage_offsets=torch.Size([0, 0]), lengths=torch.Size([1, 1])
        # )
        data = find_state_dict_object(state_dict, read_item.dest_index)
        print_0(data)
        # tensor([[0.2028]], device='cuda:0')


def test_find_state_dict_object(rank, world_size):
    h = 8
    w = 1
    tensor = torch.rand(h, w).to(rank)
    print_0(tensor)
    # tensor([[0.4280],
    #         [0.5213],
    #         [0.9812],
    #         [0.1096],
    #         [0.9135],
    #         [0.8935],
    #         [0.9513],
    #         [0.2838]], device='cuda:0')
    sharded = to_sharded(tensor, world_size)
    print_0(sharded.local_shards())
    # [
    #   Shard(
    #       tensor=tensor([[0.2028]], device='cuda:0'),
    #       metadata=ShardMetadata(
    #           shard_offsets=[0, 0],
    #           shard_sizes=[1, 1],
    #           placement=rank:0/cuda:0)
    #   )
    # ]
    state_dict = {"sharded": sharded}
    test_save_planner(state_dict)
    test_load_planner(state_dict)


def main(rank, world_size):
    setup(rank, world_size)
    test_find_state_dict_object(rank, world_size)
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
