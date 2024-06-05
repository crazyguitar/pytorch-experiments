import torch
import torch.nn as nn
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split


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
    def __init__(self, x, y, x_test, y_test, epochs=299, device="cuda"):
        self.x = x.to(device)
        self.y = y.to(device)
        self.x_test = x_test.to(device)
        self.y_test = y_test.to(device)
        self.device = device

        n_samples, n_features = x.shape
        self.n_samples = n_samples
        self.n_features = n_features
        self.lr = 0.01
        self.epochs = epochs

        self.model = LinearRegression(n_features, 1).to(device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            output = self.model(self.x)
            loss = self.loss(output, self.y)
            loss.backward()
            self.optimizer.step()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.x_test)
            accuracy = (output - self.y_test).sum().abs() / float(y_test.shape[0])
            print(f"accuracy {accuracy}")

    def run(self):
        self.train()
        self.validate()
        if self.device != "cpu":
            plot(self.model, self.x, self.y, self.x_test)


if __name__ == "__main__":
    x, y = datasets.make_regression(
        n_samples=50000, n_features=9999, noise=20, random_state=1
    )

    x, x_test, y, y_test = train_test_split(
        x, y, test_size=0.2, random_state=78
    )

    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    y = y.view(y.shape[0], 1)

    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    y_test = y_test.view(y_test.shape[0], 1)

    print("===== start =====")
    trainer = Trainer(x, y, x_test, y_test)
    trainer.run()
