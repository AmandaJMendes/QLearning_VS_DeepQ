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


if __name__ == "__main__":
    import gym
    import numpy as np

    BATCH_SIZE    = 8
    GAMMA         = 0.99  # Discount rate
    ALFA          = 0.1   # Learning rate
    EPISODES      = 10000 # Number of episodes
    MAX_STEPS     = 100   # Maximum number of steps
    HARD          = True # Controls the "is_slippery" paramater

    MAX_EPSILON   = 1     # Maximum threshold for exploration/exploitation
    MIN_EPSILON   = 0.01  # Maximum threshold for exploration/exploitation
    EPSILON_DECAY = 0.999 # Decay rate of threshold for exploration/exploitation

    replay_memory  = ReplayMemory()
    policy_network = NeuralNet()
    target_network = NeuralNet().load_state_dict(policy_network.state_dict()) 
    optimizer = torch.optim.SGD(policy_network.parameters(), lr=0.001, momentum=0.9)
    epsilon   = MAX_EPSILON

    for episode in range(EPISODES):

        if episode == (EPISODES-5): # Show enviroment in graphical interface
            env = gym.make("FrozenLake-v1", is_slippery = HARD, render_mode="human")
            state, info = env.reset()     

        for step in range(MAX_STEPS):
            if torch.rand() < epsilon: # Explore 
                action = env.action_space.sample()
            else: # Exploit
                action = torch.argmax(policy_network(state))

            next_state, reward, terminated, truncated, info = env.step(action)

            replay_memory.add_experience(state, action, reward, next_state)

            exp_batch = replay_memory.sample_batch(BATCH_SIZE)

            predicted_q = policy_network(exp_batch[:, 0])[action]
            target_q    = exp_batch[:, 2] + GAMMA*torch.max(target_network(exp_batch[:, -1]), dim = 1)

            loss = (predicted_q - target_q)**2

            loss.backward()
            optimizer.step()
       
            state = next_state

            if terminated:
                state, _ = env.reset()
                if reward: # Goal reached (reward is 1 when goal is reached and 0 otherwise)
                    wins += 1
                break

        epsilon = max(MIN_EPSILON, epsilon*EPSILON_DECAY) 

        if ((episode+1) % 1000) == 0:
            print(f"Win rate: {wins/10}%")
            wins = 0