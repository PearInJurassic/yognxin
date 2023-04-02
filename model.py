import torch
import torch.nn as nn


class testRNN(nn.Module):
    def __init__(self, history_window_length=40, feature_num=9):
        super(testRNN, self).__init__()
        self.history_window_length = history_window_length

        # self.gru = nn.GRU(history_window_length + 32, 96, num_layers=4)
        self.gru = nn.GRU(history_window_length, 96, num_layers=4)
        self.gc = nn.Sequential(
            nn.Linear(96 * 12, 60),
            nn.ReLU(),
            nn.Linear(60, 52),
            nn.ReLU(),
            nn.Linear(52, 24),
            nn.ReLU(),
            nn.Linear(24, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        temp_out, hn = self.gru(x)
        out = torch.reshape(temp_out, (batch_size, -1))
        res = self.gc(out)
        return out, hn, res
