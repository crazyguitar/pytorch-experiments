import torch
import torch.nn as nn
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def plot(model, x, y, x_test):
    import matplotlib.pyplot as plt

    if x.shape[1] != 1:
        return
    y_hat = model(x).detach().numpy()
    y_test_hat = model(x_test).detach().numpy()
    plt.plot(x.numpy(), y.numpy(), "ro")
    plt.plot(x.numpy(), y_hat, "b")
    plt.plot(x_test, y_test_hat, "go")
    plt.show()


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class Trainer:
    def __init__(self, x, y, x_test, y_test, epochs=999, device="cuda"):
        self.train_dataset = TensorDataset(x, y)
        self.test_dataset = TensorDataset(x_test, y_test)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=599,
            shuffle=False,
            #sampler=DistributedSampler(self.train_dataset),
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=599,
            shuffle=True,
        )

        self.device = device

        n_samples, n_features = x.shape
        self.n_samples = n_samples
        self.n_features = n_features
        self.lr = 0.01
        self.epochs = epochs

        self.model = LinearRegression(n_features, 1).to(device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def run_batch(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()
        return loss

    def run_epoch(self, epoch):
        total_loss = torch.zeros(2).to(self.device)
        for data, target in self.train_dataloader:
            data = data.to(self.device)
            target = target.to(self.device)
            loss = self.run_batch(data, target)
            total_loss[0] += loss.item()
            total_loss[1] += len(data)

        if epoch % 50 == 0:
            print(f"epoch: {epoch}, loss: {total_loss[0] / total_loss[1]}")

    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            self.run_epoch(epoch)
            self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()
        accuracy = torch.zeros(2).to(self.device)
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                accuracy[0] += (output - target).sum().abs()
                accuracy[1] += float(target.shape[0])

        if epoch % 50 == 0:
            print(f"accuracy {accuracy[0].item() / accuracy[1]}")


if __name__ == "__main__":
    device = "cuda"
    x, y = datasets.make_regression(
        n_samples=50000, n_features=9999, noise=20, random_state=1
    )

    x, x_test, y, y_test = train_test_split(
        x, y, test_size=0.2, random_state=78
    )

    x = torch.from_numpy(x.astype(np.float32)).to(device)
    y = torch.from_numpy(y.astype(np.float32)).to(device)
    y = y.view(y.shape[0], 1).to(device)

    x_test = torch.from_numpy(x_test.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)
    y_test = y_test.view(y_test.shape[0], 1).to(device)

    print("===== start =====")
    epochs = 1000
    trainer = Trainer(x, y, x_test, y_test, epochs=epochs, device=device)
    trainer.train()
    # plot(trainer.model, x, y, x_test)
