import gym
import numpy as np

GAMMA         = 0.99  # Discount rate
ALFA          = 0.1   # Learning rate
EPISODES      = 10000 # Number of episodes
MAX_STEPS     = 100   # Maximum number of steps
HARD          = True # Controls the "is_slippery" paramater

MAX_EPSILON   = 1     # Maximum threshold for exploration/exploitation
MIN_EPSILON   = 0.01  # Maximum threshold for exploration/exploitation
EPSILON_DECAY = 0.999 # Decay rate of threshold for exploration/exploitation

env = gym.make("FrozenLake-v1", is_slippery = HARD, render_mode="rgb_array")

n_states  = env.observation_space.n
n_actions = env.action_space.n
q_table   = np.zeros((n_states, n_actions))
epsilon   = MAX_EPSILON
state, _  = env.reset()
wins      = 0

for episode in range(EPISODES):

    if episode == (EPISODES-5): # Show enviroment in graphical interface
        env = gym.make("FrozenLake-v1", is_slippery = HARD, render_mode="human")
        state, info = env.reset()     

    for step in range(MAX_STEPS):
        if np.random.rand() < epsilon: # Explore 
            action = env.action_space.sample()
        else: # Exploit
            action = np.argmax(q_table[state, :])

        next_state, reward, terminated, truncated, info = env.step(action)

        q_table[state, action] = (1-ALFA)*q_table[state, action] + \
                                 ALFA*(reward + GAMMA*q_table[next_state, :].max())        
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

env.close()

print("--- FINAL Q-TABLE ---")
print(q_table)