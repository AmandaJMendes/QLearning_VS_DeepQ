import torch

class ReplayMemory:
    def __init__(self, N = 1000):
        self.memory = torch.empty(size = (0, 4))
        self.N = N

    def add_experience(self, state, action, reward, next_state):
        new_exp = torch.tensor([state, action, reward, next_state]).view((1, 4))
        self.memory = torch.cat([self.memory, new_exp], dim = 0)

    def sample_batch(self, batch_size = 8):
        if not self.memory.numel():
            raise Exception("Replay memory is empty!")

        if self.memory.size()[0] < batch_size:
            print(f"WARNING! REPLAY MEMORY LENGTH ({self.memory.size()[0]}) IS LESS THEN BATCH SIZE ({batch_size}).")
            return self.memory
        else:
            indices = torch.randint(low = 0, high = self.memory.size()[0], size = (batch_size,))
            batch = self.memory[indices, :]
            return batch

if __name__ == "__main__":
    replay_memory = ReplayMemory(100)

    #replay_memory.sample_batch(4)        # len(memory)=0 - Raises exception, as expected
    replay_memory.add_experience(0, 1, 1, 5)
    replay_memory.add_experience(0, 2, 2, 5)
    replay_memory.add_experience(0, 3, 3, 5)

    batch = replay_memory.sample_batch(4) # batch_size>len(memory) - Generate batch of size len(memory)
    print(batch)

    replay_memory.add_experience(0, 4, 4, 5)
    replay_memory.add_experience(0, 5, 5, 5)
    replay_memory.add_experience(0, 6, 6, 5)

    batch = replay_memory.sample_batch(4) # batch_size<len(memory) - Generate batch of size batch_size
    print(batch)