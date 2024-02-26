import torch
import gym
from memory import ReplayMemory
from network import DQN
import math

BATCH_SIZE    = 16    # Batch size
GAMMA         = 0.99  # Discount rate
ALFA          = 0.1   # Learning rate
EPISODES      = 20000 # Number of episodes
MAX_STEPS     = 100   # Maximum number of steps
HARD          = False # Controls the "is_slippery" paramater

MAX_EPSILON   = 1     # Maximum threshold for exploration/exploitation
MIN_EPSILON   = 0.1   # Maximum threshold for exploration/exploitation
EPSILON_DECAY = 0.9999 # Decay rate of threshold for exploration/exploitation

env = gym.make("FrozenLake-v1", is_slippery = HARD, render_mode="rgb_array")
state, _ = env.reset() 

n_states  = env.observation_space.n
n_actions = env.action_space.n

replay_memory  = ReplayMemory() 

states_idx = torch.arange(4, dtype = torch.float)
x_axis = states_idx.repeat_interleave(4).unsqueeze(1)
y_axis = states_idx.repeat(4).unsqueeze(1)
embedding_matrix = torch.concat([x_axis, y_axis], dim = 1)

policy_network = DQN(n_states, n_actions, embedding_matrix)
target_network = DQN(n_states, n_actions, embedding_matrix)
target_network.load_state_dict(policy_network.state_dict()) # Cloning weights from policy_network


optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-3, amsgrad=True)
epsilon   = MAX_EPSILON
wins = 0

for episode in range(EPISODES):

    if ((episode+1) % 1000) == 0: # Evaluate performance
        print(f"Win rate: {wins/10}%")
        wins = 0

    if ((episode+1) % 100) == 0:  # Update target 
        target_network.load_state_dict(policy_network.state_dict()) 

    if episode == (EPISODES-5): # Show enviroment in graphical interface
        env = gym.make("FrozenLake-v1", is_slippery = HARD, render_mode="human")
        state, _ = env.reset() # State is an integer from 0 to 15

    for step in range(MAX_STEPS):
        if torch.rand((1, )) < epsilon: # Explore 
            action = env.action_space.sample() # Integer from 0 to 3
        else:                           # Exploit
            with torch.no_grad():
                # Get action with the highest q-value from current state
                q_values = policy_network(torch.tensor(state)) # Shape [4]
                action = torch.argmax(q_values).item() # Integer from 0 to 3

        next_state, reward, done, truncated, info = env.step(action)
        replay_memory.add_experience(state, action, int(reward),
                                     next_state, int(done))

        state = next_state

        if len(replay_memory) >= BATCH_SIZE: # Train policy network
            
            exp_batch = replay_memory.sample_batch(BATCH_SIZE) # Shape [BATCH_SIZE X 5]
            # Each of the following variables has shape [BATCH_SIZE X 1]
            exp_states, exp_actions, exp_rewards, exp_next, exp_done = torch.split(exp_batch,1, dim=1)
            
            predicted_q = policy_network(exp_states.squeeze()).gather(1, exp_actions)
            
            max_q_next = torch.max(target_network(exp_next.squeeze()), dim = 1, keepdim=True).values
            # (1.0-exp_done): if the experience terminates the game, this term is set to 0.0
            target_q    = exp_rewards + \
                          (1.0-exp_done)*GAMMA*max_q_next
                          
            loss = torch.sum((predicted_q - target_q.view(BATCH_SIZE, 1))**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done or truncated:
            state, _ = env.reset()
            if reward: # Goal reached (reward is 1 when goal is reached and 0 otherwise)
                wins += 1
            break

    epsilon = max(MIN_EPSILON, epsilon*EPSILON_DECAY) # Epsilon decay

env.close()

print("--- Q-TABLE PREDICTED BY THE POLICY NETWORK ---")
print(policy_network(torch.arange(16).view(16, 1)))