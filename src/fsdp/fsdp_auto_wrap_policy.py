import os
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed as dist


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_model(device):
    return Net().to(device)


def get_fsdp_model(device):
    return FSDP(Net().to(device))


def get_fsdp_model_with_auto_wrap_policy(device):
    model = Net().to(device)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    return FSDP(model, auto_wrap_policy=auto_wrap_policy)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    if rank == 0:
        model = get_model(rank)
        print(f"model parameters: {count_parameters(model)}")
        print(model)
        model = get_fsdp_model(rank)
        print(f"fsdp model parameters: {count_parameters(model)}")
        print(model)
        model = get_fsdp_model_with_auto_wrap_policy(rank)
        print(f"fsdp wrap policy model parameters: {count_parameters(model)}")
        print(model)

    dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_nod=8 fsdp_auto_wrap_policy.py
    main()
