# ECE6254 Project - Reinforcement Learning to play Ms Pacman
import gym
import tensorflow as tf
import numpy as np

## Set Parameters
alpha = 0.01  # determines how fast we update Q, the state-value table
y = 0.99  # gamma: discount on future rewards
num_episodes = 200000  # number of episodes (complete games) we should do
eps = np.log(0.0001)/num_episodes  # chance of random exploration vs. choosing best policy (used in np.exp)

## Initialize environment and variables
# create OpenAI gym environment
#env = gym.make('FrozenLake-v0')
env = gym.make('Blackjack-v0')
# create and initialize Q (state-action) table 
#Q = np.zeros((env.observation_space.n, env.action_space.n))
Q = np.zeros((1024, 2))

# Blackjack observations are tuples of (player, dealer, usable_ace)
# these convert to an int for Q (stored as a matrix) lookup and back
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
        action = Q[tuple_to_int(s),:].argmax()
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
            
def play_ideal_strategy(env):
    '''Play a game using ideal strategy (obtained from a card bought in Las Vegas)'''
    s = env.reset()
    s1 = s
    reward = 0
    total_reward = 0
    done = False
    while(done is False):
        action = ideal_strategy(s[0], s[1], s[2])
        s1, reward, done, _ = env.step(action)
        total_reward += reward
        s = s1
    return total_reward

# fill in ideal Q
Q_ideal = np.zeros((1024,2))
for state in range(1024):
    if(state == 500):
        pass
    player, dealer, ace = int_to_tuple(state)
    if(ideal_strategy(player, dealer, ace) == 1):
        Q_ideal[state, 0] = -1
        Q_ideal[state,1] = 1
    else:
        Q_ideal[state,0] = 1
        Q_ideal[state,1] = -1
        
  
# Train Q, the state-value table by playing a bunch of games and updating
# Q via the Bellman equation
# Q function is implemented as a matrix
for e_i in range(num_episodes):
    # play through one game, updating the Q table at each step
    state = env.reset()
    done = False
    reward = 0
    while(done is False):
        # pick an action
        action = Q[tuple_to_int(state),:].argmax()
        # use a decaying random exploration rate
        if(np.random.random() < np.exp(e_i*eps)):
            #action = np.random.randint(0, env.action_space.n-1)
            action = env.action_space.sample()
        s1, reward, done, _ = env.step(action)
        # Update Q(state,action) (the state-action value function) using an
        # approximation of the incremental mean function with alpha instead
        # of 1/N
        # (see David Silver's RL lecture 4)
        if(done is True):
            # if this game is over, just count the reward with no next state
            Q[tuple_to_int(state), action] = Q[tuple_to_int(state), action] + alpha*(reward - Q[tuple_to_int(state), action])
        else:
            # if the game isn't over yet, count the reward plus expected
            # reward from the next state
            Q[tuple_to_int(state), action] = Q[tuple_to_int(state), action] + alpha*(reward + y*np.max(Q[tuple_to_int(s1), :]) - Q[tuple_to_int(state), action])
        state = s1
    

print('Mean reward per game for random agent: %f' % np.mean([play_random(env) for i in range(100000)]))
print('Mean reward for game with trained agent: %f' % np.mean([play(env, Q) for i in range(100000)]))
#print('Mean reward for play with ideal strategy agent: %f' % np.mean([play_ideal_strategy(env) for i in range(10000)]))
print('Mean reward for play with Q_ideal: %f' % np.mean([play(env, Q_ideal) for i in range(100000)]))