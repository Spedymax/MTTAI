import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, x_train, y_train, x_test, y_test):
    model = model.to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for _ in range(1000):
        opt.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train.unsqueeze(1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        pred = model(x_test)
        test_loss = criterion(pred, y_test.unsqueeze(1)).item()
        error = torch.mean(torch.abs((pred.squeeze() - y_test) / y_test)).item()

    return losses, test_loss, error, pred


def make_data(start, end):
    x = np.arange(start, end, 0.1)
    y = np.cos(x) / x - np.sin(x) / x ** 2
    z = np.sin(x / 2) + y * np.sin(x)

    inputs = torch.tensor(np.stack((x, y), axis=1), dtype=torch.float32).to(device)
    outputs = torch.tensor(z, dtype=torch.float32).to(device)

    return inputs, outputs


def test_network(model, x_train, y_train, x_test, y_test, name):
    train_loss, test_loss, rel_err, preds = train_model(model, x_train, y_train, x_test, y_test)

    print(f"\nModel: {name}")
    print(f"Final train loss: {train_loss[-1]:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Error %: {(rel_err * 100)-100:.2f}")

    return train_loss, test_loss, rel_err


x_train, y_train = make_data(20, 60)
x_test, y_test = make_data(10, 40)


class SimpleNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layer1 = nn.Linear(2, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class CascadeNet(nn.Module):
    def __init__(self, hidden, layers=1):
        super().__init__()
        self.layers = layers

        if layers == 1:
            self.layer1 = nn.Linear(2, hidden)
            self.out = nn.Linear(hidden + 2, 1)
        else:
            self.layer1 = nn.Linear(2, hidden)
            self.layer2 = nn.Linear(hidden + 2, hidden)
            self.out = nn.Linear(hidden * 2 + 2, 1)

    def forward(self, x):
        if self.layers == 1:
            h = F.relu(self.layer1(x))
            return self.out(torch.cat([x, h], dim=1))
        else:
            h1 = F.relu(self.layer1(x))
            h2 = F.relu(self.layer2(torch.cat([x, h1], dim=1)))
            return self.out(torch.cat([x, h1, h2], dim=1))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden = hidden_size
        self.n_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden).to(device)
        out, _ = self.rnn(x.unsqueeze(1), h0)
        return self.out(out.squeeze(1))


results = {}

net1 = SimpleNet(10)
results['net10'] = test_network(net1, x_train, y_train, x_test, y_test, "Simple 10")

net2 = SimpleNet(20)
results['net20'] = test_network(net2, x_train, y_train, x_test, y_test, "Simple 20")

cascade1 = CascadeNet(20)
results['casc20'] = test_network(cascade1, x_train, y_train, x_test, y_test, "Cascade 20")

cascade2 = CascadeNet(10, 2)
results['casc10x2'] = test_network(cascade2, x_train, y_train, x_test, y_test, "Cascade 10x2")

rnn1 = RNN(2, 15, 1)
results['rnn15'] = test_network(rnn1, x_train, y_train, x_test, y_test, "RNN 15")

rnn2 = RNN(2, 5, 3)
results['rnn5x3'] = test_network(rnn2, x_train, y_train, x_test, y_test, "RNN 5x3")

plt.figure(figsize=(12, 6))
names = list(results.keys())
errors = [(results[m][2] * 100)-100 for m in names]

plt.bar(names, errors)
plt.title('Errors by Model')
plt.ylabel('Error %')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()