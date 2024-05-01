import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


MNIST_URL = "https://sagemaker-example-files-prod-us-east-1.s3.amazonaws.com/datasets/image/MNIST/"
MNIST.mirrors = [MNIST_URL]


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


class Trainer:
    def __init__(self, args):
        self.epochs = args.epochs
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.lr = args.lr
        self.momentum = args.momentum
        self.data_dir = args.data
        kw = {"num_workers": 1, "pin_memory": True}
        self.train_loader = self.get_train_data_loader(
            args.batch_size, self.data_dir, **kw
        )
        self.test_loader = self.get_test_data_loader(
            args.test_batch_size, self.data_dir, **kw
        )
        self.model = self.setup_model(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )

        if dist.get_rank() == 0:
            print(self.model)

    def setup_model(self, device):
        model = Net().to(device)
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100
        )
        return FSDP(model, auto_wrap_policy=auto_wrap_policy)

    def get_train_dataset(self, train_dir, **kw):
        dataset = None
        if dist.get_rank() == 0:
            dataset = datasets.MNIST(
                train_dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )

        dist.barrier()  # prevent other ranks from accessing the data early
        if dist.get_rank() != 0:
            dataset = datasets.MNIST(
                train_dir,
                train=True,
                download=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )
        return dataset

    def get_train_data_loader(self, batch_size, train_dir, **kw):
        dataset = self.get_train_dataset(train_dir)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=DistributedSampler(dataset),
            **kw,
        )

    def get_test_dataset(self, train_dir, **kw):
        return datasets.MNIST(
            train_dir,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

    def get_test_data_loader(self, test_batch_size, train_dir, **kw):
        return torch.utils.data.DataLoader(
            self.get_test_dataset(train_dir),
            batch_size=test_batch_size,
            shuffle=True,
            **kw,
        )

    def run_batch(self, epoch, i, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
        return loss

    def run_epoch(self, epoch):
        ddp_loss = torch.zeros(2).to(self.device)
        for i, (data, target) in enumerate(self.train_loader, 1):
            data = data.to(self.device)
            target = target.to(self.device)
            loss = self.run_batch(epoch, i, data, target)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            print(f"Train Epoch: {epoch} Loss: {ddp_loss[0] / ddp_loss[1]}")

    def run_test(self):
        loss = 0
        correct = 0
        sz = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                sz += len(data)
            return loss, correct, sz

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.run_epoch(epoch)
            self.test()

    def test(self):
        self.model.eval()
        ddp_loss = torch.zeros(3).to(self.device)
        loss, correct, sz = self.run_test()
        ddp_loss[0] += loss
        ddp_loss[1] += correct
        ddp_loss[2] += sz

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            loss = ddp_loss[0] / ddp_loss[2]
            accuracy = 100.0 * ddp_loss[1] / ddp_loss[2]
            print(f"Test Average Loss: {loss:.4f}, Accuracy: {accuracy}")


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_nod=8 fsdp_mnist.py
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.01, help="learning rate")
    p.add_argument("--batch-size", type=int, default=64, help="batch size")
    p.add_argument("--test-batch-size", type=int, default=64, help="test batch size")
    p.add_argument("--epochs", type=int, default=16, help="epochs")
    p.add_argument("--momentum", type=float, default=0.5, help="momentum")
    p.add_argument("--data", type=str, default="data", help="data folder")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.manual_seed(5566)
    args = p.parse_args()
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    trainer = Trainer(args)
    trainer.train()
    dist.destroy_process_group()