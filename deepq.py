import torch
import gym
from memory import ReplayMemory
from network import DQN

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
policy_network = DQN()
target_network = DQN().load_state_dict(policy_network.state_dict()) 
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