"""LSTM classifier for sequences of normalized hand landmarks."""
import torch
from torch import nn


# Order matters: index i is output class i of the trained model
# (backend/app/ml/models/sign_lstm.pt, trained on WLASL, 10 signs).
VOCABULARY = ["Hello", "Yes", "No", "Help", "Thank You", "Emergency",
              "Please", "Sorry", "Good", "Goodbye"]


class SignLSTM(nn.Module):
    def __init__(self, input_dim: int = 63, num_classes: int = 6,
                 hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(hidden[-1])
