
import gym
import tensorflow as tf
import numpy as np

## Set Parameters
lr = 0.8
y = 0.99  # gamma: discount on future rewards
num_episodes = 4000  # number of episodes (complete plays) we should do
eps = np.log(0.1)/num_episodes  # change of random exploration vs. choosing best policy

## Initialize environment and variables
# create OpenAI gym environment
#env = gym.make('FrozenLake-v0')
env = gym.make('Blackjack-v0')
# create and initialize Q (state-action) table 
#Q = np.zeros((env.observation_space.n, env.action_space.n))
Q = np.zeros((1024, 2))

tuple_to_int = lambda t: (t[0] << 5) + (t[1] << 1) + int(t[2])
int_to_tuple = lambda i: ((i >> 5), ((i >> 1) & 0xf), ((i & 1 == 1)))

def play(env, Q):
    '''Play a game using policy Q and return the reward'''
    s = env.reset()
    s1 = s
    reward = 0
    total_reward = 0
    done = False
    while(done is False):
        action = Q[state,:].argmax()
        s1, reward, done, _ = env.step(action)
        total_reward += reward
        s = s1
    return total_reward
    
    
def play_random(env):
    '''Play a game using a random policy and return the reward'''
    s = env.reset()
    s1 = s
    reward = 0
    total_reward = 0
    done = False
    while(done is False):
        action = env.action_space.sample()
        s1, reward, done, _ = env.step(action)
        total_reward += reward
        s = s1
    return total_reward
    
    
def ideal_strategy(player, dealer, ace_11):
    '''Return 0 if the ideal play is to stand, 1 if hit'''
    if(ace_11 == False):
        if(player >= 4 and player <= 11):
            return 1
        elif(player == 12):
            if(dealer <= 3):
                return 1
            elif(dealer <= 6):
                return 0
            else:
                return 1
        elif(player >= 13 and player <= 16):
            if(dealer <= 6):
                return 0
            else:
                return 1
        else:
            return 0
    else:
        # soft hands
        if(player <= 17):
            return 1
        elif(player == 18):
            if(dealer <= 8):
                return 0
            else:
                return 1
        else:
            return 0
    
# fill in ideal Q
Q_ideal = np.zeros((1024,2))
for state in range(1024):
    player, dealer, ace = int_to_tuple(state)
    if(ideal_strategy(player, dealer, ace) == 1):
        Q[state,0] = -1
        Q[state,1] = 1
    else:
        Q[state,0] = 1
        Q[state,1] = -1
    
    
    
    

for e_i in range(num_episodes):
    # play through one game, updating the Q table at each step
    state = tuple_to_int(env.reset())
    done = False
    reward = 0
    while(done is False):
        # pick an action
        action = Q[state,:].argmax()
        if(np.random.random() < np.exp(-e_i*eps)):
            action = env.action_space.sample()
        s1, reward, done, _ = env.step(action)
        s1 = tuple_to_int(s1)
        Q[state, action] = Q[state, action] + lr*(reward + y*np.max(Q[s1, :]) - Q[state,action])
        state = s1
    #if(not np.isclose(reward, 0.0)):
    #    print('Final reward: %f' % reward)