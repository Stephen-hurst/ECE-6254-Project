# ECE6254 Project - Reinforcement Learning to play Ms Pacman
import gym
import tensorflow as tf
import numpy as np
import sklearn.svm

## Set Parameters
alpha = 0.01  # determines how fast we update Q, the state-value table
y = 0.95  # gamma: discount on future rewards
num_episodes = 2000000  # number of episodes (complete games) we should do
eps = np.log(0.0001)/num_episodes  # chance of random exploration vs. choosing best policy (used in np.exp)

## Initialize environment and variables
# create OpenAI gym environment
#env = gym.make('FrozenLake-v0')
env = gym.make('Blackjack-v0')

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
    
def train_Q_incrementally(Q):
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
    return Q
    
def train_Q_batch(Q):
    '''Train Q using a batch method. Q should already be initialized so
    Q.predict works.'''
    # Train using a batch of several games at once
    # Use experience replay and fixed-Q targets to update Q
    # experience replay: keep a replay history (s, a, r, s') and train using
    # samples from that history
    # fixed-Q: use a previous version of Q in each batch run
    OBSERVATIONS = 3
    ACTIONS = 2
    batch_size = 100 # how many games do we play per batch
    num_batches = 1000
    replay_memory = np.zeros((0, OBSERVATIONS*2+3))  # history of (s, a, r, s') experiences
    for batch_i in range(num_batches):
        batch_memory = []
        for e_i in range(batch_size):
            s = env.reset()
            r = 0
            done = False
            while(done is False):
                # create a matrix with the current state and all actions
                player, dealer, ace = s
                all_actions = np.tile(np.array([[player,dealer,ace,0]]), (ACTIONS,1))
                all_actions[:,-1] = np.arange(ACTIONS)
                # Choose the action that maximizes value
                action = Q.predict(all_actions).argmax()
                # use a decaying random exploration rate to sometimes try
                # random actions
                if(np.random.random() < np.exp(e_i*eps)):
                    action = env.action_space.sample()
                s1, r, done, _ = env.step(action)
                s1_player, s1_dealer, s1_ace = s1
                batch_memory.append(np.array([player, dealer, ace, action, r, done, s1_player, s1_dealer, s1_ace]))
        # Update Q using the current replay memory
        replay_memory = np.concatenate((replay_memory, np.array(batch_memory)))
        # Sample from the current replay memory
        # calculate targets using the current Q and use those targets to re-train Q
        X_sample = replay_memory[np.random.choice(replay_memory.shape[0], size=min([replay_memory.shape[0], 1000]), replace=False),:]
        # S,A (state we were in and action we took when we made this observation)
        S_A = X_sample[:,0:(OBSERVATIONS+1)] 
        # S' (state we moved to next)
        S1 = X_sample[:,-OBSERVATIONS:]
        # We need to calculate new targets to use as training values
        # First, we need to calculate max_a' for all actions a' from state s'
        # repeat S1 for all possible actions
        S1_A = np.tile(S1, (ACTIONS,1))
        # add a column with all possible actions repeated for every sample
        S1_A = np.concatenate((S1_A, np.repeat(np.arange(ACTIONS), X_sample.shape[0]).reshape((S1_A.shape[0],1))), axis=1)
        # predict q(s',a') for all actions a'
        all_action_values = Q.predict(S1_A)
        # Collapse into a 2-d array where rows are samples and columns are actions
        all_action_values = all_action_values.reshape((X_sample.shape[0],ACTIONS), order='F')
        reward_column = X_sample[:,OBSERVATIONS+1]
        done_column = X_sample[:,OBSERVATIONS+2]
        argmax_s1 = (1-done_column)*np.max(all_action_values, 1)        
        # training values are the TD-targets
        y_train = alpha*(reward_column + y*argmax_s1 - Q.predict(S_A))
        Q.fit(S_A, y_train)
        #Q[tuple_to_int(state), action] = Q[tuple_to_int(state), action] + alpha*(reward + y*np.max(Q[tuple_to_int(s1), :]) - Q[tuple_to_int(state), action])
    return Q


# Incremental training using a lookup table
# create and initialize Q (state-action) table 
Q_random = np.random.random((1024,2))*2 - 1
print('Mean reward per game for random agent (Q before training): %f' % np.mean([play(env,Q_random) for i in range(100000)]))
Q = train_Q_incrementally(Q_random)
#print('Mean reward per game for random agent: %f' % np.mean([play_random(env) for i in range(100000)]))
print('Mean reward for game with trained agent: %f' % np.mean([play(env, Q) for i in range(100000)]))
#print('Mean reward for play with ideal strategy agent: %f' % np.mean([play_ideal_strategy(env) for i in range(10000)]))
print('Mean reward for play with Q_ideal: %f' % np.mean([play(env, Q_ideal) for i in range(100000)]))

# # Batch training using linear regression
# # Initialize Q function randomly
# Q = sklearn.svm.SVR()
# Q.fit(np.random.random((1, 4)), np.random.random(1))
# Q = train_Q_batch(Q)