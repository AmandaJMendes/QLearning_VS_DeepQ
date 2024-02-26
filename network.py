import torch

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_states, n_actions, embedding_matrix):
        super(DQN, self).__init__()

        self.embedding_matrix = embedding_matrix

        self.fc1        = nn.Linear(2, 128)
        self.fc2        = nn.Linear(128, 64)
        self.fc3        = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.embedding(x, self.embedding_matrix)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    states_idx = torch.arange(4, dtype = torch.float)
    x_axis = states_idx.repeat_interleave(4).unsqueeze(1)
    y_axis = states_idx.repeat(4).unsqueeze(1)
    embedding_matrix = torch.concat([x_axis, y_axis], dim = 1)
    print(embedding_matrix)

    net = DQN(16, 4, embedding_matrix)
    states = torch.tensor([0, 4, 1, 5, 11])
    print(F.embedding(states, embedding_matrix))
    
    out = net(states)
    print(out)