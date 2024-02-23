import torch

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        #self.embeddings = nn.Embedding(num_embeddings = n_states, embedding_dim = 4)
        self.fc1        = nn.Linear(1, 128)
        self.fc2        = nn.Linear(128, 64)
        self.fc3        = nn.Linear(64, n_actions)

    def forward(self, x):
        #x = self.embeddings(x)
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
