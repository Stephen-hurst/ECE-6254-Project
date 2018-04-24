# ECE6254 Project - Reinforcement Learning to play Ms Pacman
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm
import sys
sys.path.append('ECE-6254-Project')
from lookup_table import LookupTable
from game_history import GameHistory
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input

## Set Parameters
alpha = 0.01  # determines how fast we update Q, the state-value table
gamma = 0.95  # gamma: discount on future rewards
num_episodes = 2000000  # number of episodes (complete games) we should do
eps = np.log(0.0001)  # chance of random exploration vs. choosing best policy (used in np.exp)

## Initialize environment and variables
# create OpenAI gym environment
#env = gym.make('FrozenLake-v0')
env = gym.make('Blackjack-v0')

# Blackjack observations are tuples of (player, dealer, usable_ace)
# these convert to an int for Q (stored as a matrix) lookup and back
tuple_to_int = lambda t: (t[0] << 5) + (t[1] << 1) + int(t[2])
int_to_tuple = lambda i: ((i >> 5), ((i >> 1) & 0xf), ((i & 1 == 1)))

def best_action(Q, X, num_actions):
    '''Return the best action for a number of states X using policy Q'''
    # repeat each row of X for every possible action
    X_repeat = np.repeat(X, num_actions, 0)
    # add a column for actions
    X_repeat = np.concatenate((X_repeat, np.zeros((X_repeat.shape[0], 1))), axis=1)
    # fill in sequential actions
    X_repeat[:, -1] = np.tile(np.arange(num_actions), X.shape[0])
    # predict everything
    predict = Q.predict(X_repeat)
    # reshape so rows are the original X rows and columns are predictions for 
    # each action
    predict = predict.reshape((X.shape[0], num_actions))
    # return the max action
    return predict.argmax(axis=1)
    

def best_action_value(Q, X, num_actions):
    '''Return the best action value for a number of states X using policy Q'''
    # repeat each row of X for every possible action
    X_repeat = np.repeat(X, num_actions, 0)
    # add a column for actions
    X_repeat = np.concatenate((X_repeat, np.zeros((X_repeat.shape[0], 1))), axis=1)
    # fill in sequential actions
    X_repeat[:, -1] = np.tile(np.arange(num_actions), X.shape[0])
    # predict everything
    predict = Q.predict(X_repeat)
    # reshape so rows are the original X rows and columns are predictions for 
    # each action
    predict = predict.reshape((X.shape[0], num_actions))
    # return the max action
    return predict.max(axis=1)
    

def play(env, Q):
    '''Play a game using policy Q and return the reward'''
    s = env.reset()
    s1 = s
    reward = 0
    total_reward = 0
    done = False
    while(done is False):
        if(type(Q) == np.ndarray):
            action = Q[tuple_to_int(s),:].argmax()
        else:
            action = best_action(Q, np.array([[int(s[0]), int(s[1]), int(s[2])]]), env.action_space.n)[0]
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
    
def train_Q_incrementally(Q, measure_every=None, game_means=None):
    # Train Q, the state-value table by playing a bunch of games and updating
    # Q via the Bellman equation
    # Q function is implemented as an arbitrary model with .fit and .predict
    # functions
    OBSERVATIONS = 3
    ACTIONS = 2
    SAMPLES_TO_TRAIN_ON = 1
    for e_i in range(num_episodes):
        if(measure_every != None and ((e_i % measure_every) == 0)):
            # measure current model performance
            mean_performance = np.mean([play(env, Q) for i in range(10000)])
            game_means.append(mean_performance)
        # play through one game, updating the Q table at each step
        player, dealer, ace = env.reset()
        done = False
        reward = 0
        while(done is False):
            # pick an action
            action = best_action(Q, np.array([[player, dealer, ace]]), ACTIONS)[0]
            # use a decaying random exploration rate
            if(np.random.random() < np.exp(e_i/num_episodes*eps)):
                #action = np.random.randint(0, env.action_space.n-1)
                action = env.action_space.sample()
            s1, reward, done, _ = env.step(action)
            # Update Q(state,action) (the state-action value function) using an
            # approximation of the incremental mean function with alpha instead
            # of 1/N
            # (see David Silver's RL lecture 4)
            S_A = np.array([[player, dealer, ace, action]])
            player1, dealer1, ace1 = s1
            S_1 = np.array([[player1, dealer1, ace1]])
            Q_S_A = Q.predict(S_A)
            if(done is True):
                # if this game is over, just count the reward with no next state
                Q.fit(S_A, Q_S_A + alpha*(reward - Q_S_A))
            else:
                # if the game isn't over yet, count the reward plus expected
                # reward from the next state
                Q.fit(S_A, Q_S_A + alpha*(reward + gamma*best_action_value(Q, S_1, ACTIONS) - Q_S_A))
            player, dealer, ace = s1
    # update the last measure of model performance
    if(measure_every != None):
        # measure current model performance
        mean_performance = np.mean([play(env, Q) for i in range(10000)])
        game_means.append(mean_performance)
    return Q
    
def train_Q_batch(Q, alpha=1.0, measure_every=None, game_means=None, batch_size=100, num_batches=2000, samples_to_train_on=1000):
    '''Train Q using a batch method. Q should already be initialized so
    Q.predict works.
    If measure_every is set to some value, this function will measure the model's
    performance with a period of that many games, and put the mean value
    in game_means (which should be an array)'''
    # Train using a batch of several games at once
    # Use experience replay and fixed-Q targets to update Q
    # experience replay: keep a replay history (s, a, r, s') and train using
    # samples from that history
    # fixed-Q: use a previous version of Q in each batch run
    OBSERVATIONS = 3
    ACTIONS = 2
    replay_memory = GameHistory(OBSERVATIONS*2+3, 50000) # history of (s, a, r, s') experiences
    for batch_i in range(num_batches):
        print('Batch %d' % batch_i)
        if(measure_every != None and (((batch_i*batch_size) % measure_every) == 0)):
            # measure current model performance
            mean_performance = np.mean([play(env, Q) for i in range(10000)])
            game_means.append(mean_performance)
            print('Mean game performance: %f' % mean_performance)
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
                if(np.random.random() < np.exp(batch_i/num_batches*eps)):
                    action = env.action_space.sample()
                s1, r, done, _ = env.step(action)
                s1_player, s1_dealer, s1_ace = s1
                replay_memory.add(np.array([[player, dealer, ace, action, r, done, s1_player, s1_dealer, s1_ace]]))
                s = s1
        # Update Q using the current replay memory
        # Sample from the current replay memory
        # calculate targets using the current Q and use those targets to re-train Q
        X_sample = replay_memory.sample(samples_to_train_on)
        # # use the last batch of episodes
        # X_sample = np.array(batch_memory)
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
        max_a_s1 = (1-done_column)*np.max(all_action_values, 1)        
        # training values are the TD-targets
        Q_predict = Q.predict(S_A).reshape(S_A.shape[0],)
        if('train_on_batch' in dir(Q)):
            y_train = reward_column + gamma*max_a_s1
            Q.train_on_batch(S_A, y_train)
        else:
            y_train = Q_predict + alpha*(reward_column + gamma*max_a_s1 - Q_predict)
            if('partial_fit' in dir(Q)):
                Q.partial_fit(S_A, y_train)
            else:
                Q.fit(S_A, y_train)
    # update the last measure of model performance
    if(measure_every != None):
        # measure current model performance
        mean_performance = np.mean([play(env, Q) for i in range(10000)])
        game_means.append(mean_performance)
        print('Mean game performance: %f' % mean_performance)
    return Q


# create a matrix of all input states for blackjack
X = []
for player in range(2,22):
    for dealer in range(1,22):
        for ace in range(2):
            for action in range(2):
                X.append([player, dealer, ace, action])
X = np.array(X)

# ## Incremental training
# # create and initialize Q (state-action value) function
Q_random = LookupTable()
Q_random.fit(X, np.random.random(X.shape[0]))
print('Mean reward per game for random agent (Q before training): %f' % np.mean([play(env,Q_random) for i in range(100000)]))
incremental_game_means = []
# Q = train_Q_incrementally(Q_random, measure_every=10000, game_means=incremental_game_means)
# print('Mean reward for game with trained agent: %f' % np.mean([play(env, Q) for i in range(100000)]))
# print('Mean reward for play with Q_ideal (Vegas blackjack strategy card): %f' % np.mean([play(env, Q_ideal) for i in range(100000)]))
# # Fit an SVM regression model to the lookup table version of Q to see what's
# # the best we can do with an SVM
# Q_SVM = sklearn.svm.SVR()
# Q_SVM.fit(X, Q.predict(X))
# print('Mean reward for best-fit SVM regression (SVR fit to the lookup table): %f' % np.mean([play(env, Q_SVM) for i in range(100000)]))
# 
#     
## Batch training
# # Batch training using lookup table
# Q_table_game_means = []  # global table to store game means (a metric of model performance) during training
# Q = LookupTable()
# Q = train_Q_batch(Q, alpha=0.01, num_batches=500, measure_every=10000, game_means=Q_table_game_means)
# print('Mean reward for batch-trained lookup table: %f' % np.mean([play(env, Q) for i in range(100000)]))
# # convert the trained output from lookup table to an array so we can test regression
# y = Q.predict(X)

# # Batch training using linear regression
# linear_game_means = []
# Q_linear = sklearn.linear_model.LinearRegression()
# Q_linear.fit(np.random.random((1000,4)), np.random.random(1000))
# Q_linear = train_Q_batch(Q_linear, alpha=0.01, measure_every=10000, game_means=linear_game_means)
# print('Mean reward for batch-trained LR: %f' % np.mean([play(env, Q_linear) for i in range(100000)]))

# # Batch training using SVR regression
# svr_game_means = []
# Q_SVR = sklearn.svm.SVR()
# Q_SVR.fit(np.random.random((1000, 4)), np.random.random(1000))
# #Q_SVR = train_Q_batch(Q_SVR, alpha=0.001, measure_every=10000, game_means=svr_game_means)
# Q_SVR = train_Q_batch(Q_SVR, alpha=0.001, samples_to_train_on=10000)
# print('Mean reward for batch-trained SVR: %f' % np.mean([play(env, Q_SVR) for i in range(100000)]))

# # Linear regression model (stochastic gradient descent update)
# Q = sklearn.linear_model.SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.01)
# Q.fit(np.array([[0,0,0,0]]), np.array([0]))
# Q = train_Q_batch(Q, alpha=1.0)
# print(np.mean([play(env, Q) for i in range(10000)]))
 
# Keras deep neural net
dnn_model = Sequential()
dnn_model.add(Dense(units=2048, input_dim=X.shape[1], activation='relu'))
dnn_model.add(Dense(units=100, activation='relu'))
dnn_model.add(Dense(units=1, activation='tanh'))
dnn_model.compile(optimizer=keras.optimizers.adagrad(), loss='mean_squared_error')
dnn_game_means = []
dnn_model = train_Q_batch(dnn_model, alpha=1.0, measure_every=10000, game_means=dnn_game_means)