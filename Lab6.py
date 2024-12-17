import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Simple dataset class
class StockDataset(Dataset):
    def __init__(self, data, window_size=5):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# Simple neural network
class SimplePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def main():
    try:
        # Read data
        df = pd.read_csv('HistoricalData_1733597016202.csv')

        # Check if the required column exists
        if 'Close/Last' not in df.columns:
            raise ValueError("Column 'Close/Last' not found in the CSV file")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Get just the closing prices and reverse (oldest first)
    prices = df['Close/Last'].values[::-1]  # Data is already in numeric format

    # Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Create dataset
    window_size = 10
    dataset = StockDataset(prices_scaled, window_size)
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Create model
    model = SimplePredictor(window_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    epochs = 1000
    train_losses = []

    print("Training started...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    # Test the model
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            predictions.extend(output.numpy())
            actuals.extend(batch_y.numpy())

    # Scale back to original prices
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('NASDAQ Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Predict next day's price
    last_window = torch.FloatTensor(prices_scaled[-window_size:]).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        next_day_scaled = model(last_window)
        next_day_price = scaler.inverse_transform(next_day_scaled.numpy().reshape(-1, 1))[0][0]

    current_price = prices[-1]
    price_change = next_day_price - current_price
    percent_change = (price_change / current_price) * 100

    print("\nPrediction for next day:")
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted price: ${next_day_price:.2f}")
    print(f"Predicted change: ${price_change:.2f} ({percent_change:.2f}%)")



if __name__ == "__main__":
    main()