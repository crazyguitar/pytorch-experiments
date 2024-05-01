import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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
        torch.manual_seed(args.seed)
        self.data = Path(__file__).parent / "data"
        self.train_batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.gamma = args.gamma
        self.epochs = args.epochs
        self.log_interval = args.log_interval
        self.save_model = args.save_model
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = dist.get_rank()
        self.device = self.setup_device(self.local_rank)
        self.world_size = dist.get_world_size()
        self.transform = self.setup_transform()
        self.train_dataset = self.setup_train_dataset(self.data, self.transform)
        self.test_dataset = self.setup_test_dataset(self.data, self.transform)
        self.train_dataloader = self.setup_dataloader(self.train_dataset, self.train_batch_size)
        self.test_dataloader = self.setup_dataloader(self.test_dataset, self.test_batch_size)
        self.model = self.setup_model(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.optimizer = self.setup_optimizer(self.model, self.lr)
        self.scheduler = self.setup_scheduler(self.optimizer, 1, self.gamma)

    def setup_device(self, local_rank):
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    def setup_transform(self):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup_train_dataset(self, data, transform):
        dataset = None
        if dist.get_rank() == 0:
            dataset = datasets.MNIST(data, train=True, download=True, transform=transform)
        dist.barrier()  # prevent other ranks from accessing the data early
        if dist.get_rank() != 0:
            dataset = datasets.MNIST(data, train=True, download=False, transform=transform)
        return dataset

    def setup_test_dataset(self, data, transform):
        dataset = None
        if dist.get_rank() == 0:
            dataset = datasets.MNIST(data, train=False, download=True, transform=transform)
        dist.barrier()
        if dist.get_rank() != 0:
            dataset = datasets.MNIST(data, train=False, download=False, transform=transform)
        return dataset

    def setup_dataloader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset),
        )

    def setup_model(self, device):
        return Net().to(device)

    def setup_optimizer(self, model, lr):
        return optim.Adadelta(model.parameters(), lr=lr)

    def setup_scheduler(self, optimizer, step_size, gamma):
        return StepLR(optimizer, step_size, gamma=gamma)

    def run_batch(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            loss = self.run_batch(data, target)
            if self.rank == 0 and batch_idx % self.log_interval == 0:
                size = batch_idx * len(data)
                total = len(self.train_dataloader.dataset)
                print(f"Train Epoch: {epoch} [{size}/{total}] Loss: {loss}")

    def run_batch_eval(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        output = self.model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return loss, correct

    def run_eval(self, epoch):
        self.model.eval()
        total_loss = torch.zeros(3).to(self.device)
        with torch.no_grad():
            for data, target in self.test_dataloader:
                l, c = self.run_batch_eval(data, target)
                total_loss[0] += l
                total_loss[1] += c
                total_loss[2] += target.shape[0]

        # collect all losses
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        correct = total_loss[1]
        samples = total_loss[2]
        loss = total_loss[0] / total_loss[2]
        accuracy = 100.0 * correct / samples
        if self.rank == 0:
            print(f"[Test] Average loss: {loss} Accuracy: {correct}/{samples} ({accuracy:.6f}%)")

    def run(self):
        for epoch in range(1, self.epochs + 1):
            self.run_epoch(epoch)
            self.run_eval(epoch)
            self.scheduler.step()

    def save(self):
        if self.rank == 0 and self.save_model:
            path = Path(__file__).parent / "mnist_cnn.pt"
            torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_nod=8 ddp_mnist.py

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        metavar="N",
        help="number of epochs to train (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=128,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("please run the script on cuda machines")

    try:
        dist.init_process_group(backend="nccl")
        trainer = Trainer(args)
        trainer.run()
        trainer.save()
    finally:
        dist.destroy_process_group()
