import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),   #benign/malicious
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
