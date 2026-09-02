import torch
import torch.nn as nn

from src.configuration.config import LOCKED_CONFIG

'''
Input_Dim -> 9 Features
'''
class AttentionMIL(nn.Module):
    def __init__(self, input_dim=8,hidden_dim=64, attention_dim=32, dropout=LOCKED_CONFIG["dropout"]):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

        self.classifier = nn.Linear(hidden_dim, 1)


    def forward(self,x, mask):
        h =  self.encoder(x)

        attention_scores = self.attention(h).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=1)

        weighted_sum = torch.sum(
            h * attention_weights.unsqueeze(-1),
            dim = 1
        )

        logits = self.classifier(weighted_sum).squeeze(-1)
        return logits, attention_weights