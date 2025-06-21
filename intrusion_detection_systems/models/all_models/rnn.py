import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, sequence_length):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
