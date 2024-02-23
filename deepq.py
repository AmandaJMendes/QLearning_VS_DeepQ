import torch
import gym
from memory import ReplayMemory
from network import DQN

BATCH_SIZE    = 4
GAMMA         = 0.99  # Discount rate
ALFA          = 0.1   # Learning rate
EPISODES      = 10000    # Number of episodes
MAX_STEPS     = 100   # Maximum number of steps
HARD          = False # Controls the "is_slippery" paramater

MAX_EPSILON   = 1     # Maximum threshold for exploration/exploitation
MIN_EPSILON   = 0.01  # Maximum threshold for exploration/exploitation
EPSILON_DECAY = 0.999 # Decay rate of threshold for exploration/exploitation

env = gym.make("FrozenLake-v1", is_slippery = HARD, render_mode="rgb_array")
n_states  = env.observation_space.n
n_actions = env.action_space.n
state, info = env.reset()    

replay_memory  = ReplayMemory()
policy_network = DQN(n_states, n_actions)
target_network = DQN(n_states, n_actions)
target_network.load_state_dict(policy_network.state_dict()) 

optimizer = torch.optim.AdamW(policy_network.parameters(), lr=1e-4, amsgrad=True)
epsilon   = MAX_EPSILON

for episode in range(EPISODES):
    wins = 0

    if ((episode+1) % 1000) == 0:
        print(f"Win rate: {wins/10}%")
        print("EPSILON: ", epsilon)
        wins = 0
        
    if ((episode+1) % 100) == 0:
        target_network.load_state_dict(policy_network.state_dict()) 

    if episode == (EPISODES-5): # Show enviroment in graphical interface
        env = gym.make("FrozenLake-v1", is_slippery = HARD, render_mode="human")
        state, info = env.reset()     

    for step in range(MAX_STEPS):
        if torch.rand((1, )) < epsilon: # Explore 
            action = env.action_space.sample()
        else: # Exploit
            with torch.no_grad():
                action = torch.argmax(policy_network(torch.tensor(state).view(1, 1))).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        replay_memory.add_experience(state, action, int(reward), next_state)

        state = next_state

        if len(replay_memory) >= BATCH_SIZE: # Train policy network
            
            exp_batch = replay_memory.sample_batch(BATCH_SIZE)
            predicted_q = policy_network(exp_batch[:, [0]]).gather(1, exp_batch[:, [1]])
            target_q    = exp_batch[:, [2]] + \
                          GAMMA*torch.max(target_network(exp_batch[:, [-1]]), dim = 1)[0]

            loss = torch.sum((predicted_q - target_q)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if terminated:
            state, _ = env.reset()
            if reward: # Goal reached (reward is 1 when goal is reached and 0 otherwise)
                wins += 1
            break

    epsilon = max(MIN_EPSILON, epsilon*EPSILON_DECAY) 



env.close()

print(policy_network(torch.arange(16).view(16, 1)))

# print(policy_network.state_dict())
# print(target_network.state_dict())

