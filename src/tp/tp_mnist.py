import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from pathlib import Path
from torch.distributed._tensor import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


MNIST_URL = "https://sagemaker-example-files-prod-us-east-1.s3.amazonaws.com/datasets/image/MNIST/"
MNIST.mirrors = [MNIST_URL]


def print_0(*a, **kw):
    if dist.get_rank() == 0:
        print(*a, **kw)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Trainer:
    def __init__(self, args):
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
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
        self.load()
        print_0(self.model)

    def setup_model(self, device):
        model = Net().to(device)
        parallelize_plan = {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
        }
        mesh_shpe = (dist.get_world_size(),)
        tp_mesh = init_device_mesh("cuda", mesh_shpe)
        model = parallelize_module(model, tp_mesh, parallelize_plan)
        print_0(f"total parameters: {count_parameters(model)}")
        return model

    def get_train_dataset(self, train_dir, **kw):
        dataset = None
        if dist.get_rank() == 0:
            dataset = datasets.MNIST(
                train_dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )

        dist.barrier()  # prevent other ranks from accessing the data early
        if dist.get_rank() != 0:
            dataset = datasets.MNIST(
                train_dir,
                train=True,
                download=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )
        return dataset

    def get_train_data_loader(self, batch_size, train_dir, **kw):
        dataset = self.get_train_dataset(train_dir)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **kw,
        )

    def get_test_dataset(self, train_dir, **kw):
        return datasets.MNIST(
            train_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
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
        loss = None
        for i, (data, target) in enumerate(self.train_loader, 1):
            data = data.to(self.device)
            target = target.to(self.device)
            loss = self.run_batch(epoch, i, data, target)

        print_0(f"Train Epoch: {epoch} Loss: {loss}")

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
            if epoch % 3 == 0:
                self.save()

    def test(self):
        self.model.eval()
        ddp_loss = torch.zeros(3).to(self.device)
        loss, correct, sz = self.run_test()
        ddp_loss[0] += loss
        ddp_loss[1] += correct
        ddp_loss[2] += sz

        if dist.get_rank() == 0:
            loss = ddp_loss[0] / ddp_loss[2]
            accuracy = 100.0 * ddp_loss[1] / ddp_loss[2]
            print_0(f"Test Average Loss: {loss:.4f}, Accuracy: {accuracy}")

    def save(self):
        if dist.get_rank() == 0:
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        dist.barrier()
        model_state_dict = self.model.state_dict()
        state_dict = {
            "model": model_state_dict,
        }
        dcp.save(state_dict, checkpoint_id=str(self.checkpoint_dir))
        print_0("==> save checkpoint success")

    def load(self):
        if not self.checkpoint_dir.exists():
            return

        model_state_dict = self.model.state_dict()
        state_dict = {
            "model": model_state_dict,
        }
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=self.checkpoint_dir,
        )
        self.model.load_state_dict(state_dict["model"])
        print_0("==> load checkpoint success")


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_nod=8 tp_mnist.py
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
