import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.z_dim =z_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim,300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, self.z_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 75),
            nn.ReLU(),
            nn.Linear(75, 150),
            nn.ReLU(),
            nn.Linear(150, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, self.input_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        res = x
        res = self.encoder(res)
        res = self.decoder(res)
        return res