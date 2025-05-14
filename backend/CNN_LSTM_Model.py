import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_dim=86, hidden_dim=64, lstm_layers=1, dropout=0.3):
        super(CNNLSTMClassifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.cnn(x)         # (B, 128, T)
        x = x.permute(0, 2, 1)  # (B, T, 128)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.classifier(out)
        return out.squeeze(1)
