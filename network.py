import torch

import torch.nn as nn
import torch.nn.functional as F
embedding_matrix  = torch.tensor([[0.0, 0.0],
                                  [0.0, 1.0],
                                  [0.0, 2.0],
                                  [0.0, 3.0],
                                  [1.0, 0.0],
                                  [1.0, 1.0],
                                  [1.0, 2.0],
                                  [1.0, 3.0],
                                  [2.0, 0.0],
                                  [2.0, 1.0],
                                  [2.0, 2.0],
                                  [2.0, 3.0],
                                  [3.0, 0.0],
                                  [3.0, 1.0],
                                  [3.0, 2.0],
                                  [3.0, 3.0]])
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        #self.embeddings = nn.Embedding(num_embeddings = n_states, embedding_dim = 4)
        self.fc1        = nn.Linear(2, 128)
        self.fc2        = nn.Linear(128, 64)
        self.fc3        = nn.Linear(64, n_actions)

    def forward(self, x):
        #x = self.embeddings(x)
        x = F.embedding(x, embedding_matrix)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = DQN(16, 4)
    states = torch.tensor([0, 4, 1, 5, 11])
    print(F.embedding(states, embedding_matrix))
    out = net(states)
    print(out)