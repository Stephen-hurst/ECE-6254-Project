# ECE6254 Project - Reinforcement Learning to play Ms Pacman
import gym
import tensorflow as tf
import numpy as np
import sklearn.svm
import sys
sys.path.append('ECE-6254-Project')
from lookup_table import LookupTable

## Set Parameters
alpha = 0.01  # determines how fast we update Q, the state-value table
y = 0.95  # gamma: discount on future rewards
num_episodes = 2000000  # number of episodes (complete games) we should do
eps = np.log(0.0001)/num_episodes  # chance of random exploration vs. choosing best policy (used in np.exp)

## Initialize environment and variables
# create OpenAI gym environment
env = gym.make('Pong-ram-v0')
# only use memory locations that change through the game (predetermined by
# calling find_changing_memory_locations)
memory_that_changes = np.array([64, 2, 67, 4, 69, 8, 9, 10, 11, 12, 13, 73, 15, 17, 18, 19, 20, 21, 58, 49, 50, 51, 54, 56, 121, 122, 60])
OBSERVATIONS = memory_that_changes.shape[0]
ACTIONS = 6


def normalize_state(state):
    '''Normalize a state directly from the OpenAI gym environment, i.e.
    pick out only useful features, subtract the mean, and divide by variance'''
    state = state[memory_that_changes]
    state = np.array(state, dtype='float64')
    state -= 128.
    state /= 256.
    return state


def best_action(Q, X, num_actions=ACTIONS):
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
    

def best_action_value(Q, X, num_actions=ACTIONS):
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
    
    
def train_Q_batch(Q):
    '''Train Q using a batch method. Q should already be initialized so
    Q.predict works.'''
    # Train using a batch of several games at once
    # Use experience replay and fixed-Q targets to update Q
    # experience replay: keep a replay history (s, a, r, s') and train using
    # samples from that history
    # fixed-Q: use a previous version of Q in each batch run
    SAMPLES_TO_TRAIN_ON = 1000
    batch_size = 100  # how many games do we play per batch
    num_batches = 2000
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
                s = s1
        # Update Q using the current replay memory
        replay_memory = np.concatenate((replay_memory, np.array(batch_memory)))
        # Sample from the current replay memory
        # calculate targets using the current Q and use those targets to re-train Q
        X_sample = replay_memory[np.random.choice(replay_memory.shape[0], size=min([replay_memory.shape[0], SAMPLES_TO_TRAIN_ON]), replace=False),:]
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
        Q_predict = Q.predict(S_A)
        y_train = Q_predict + alpha*(reward_column + y*max_a_s1 - Q_predict)
        Q.fit(S_A, y_train)
    return Q
    
    
def find_changing_memory_locations(games_to_play=1000):
    '''Find memory locations that change by playing a bunch of games and
    looking for memory that changes'''
    state0 = env.reset()
    reward = 0.
    done = False
    
    memory_that_changes = set()
    
    for game_i in range(games_to_play):
        while(done == False):
            state1, reward, done, _ = env.step(env.action_space.sample())
            memory_that_changes = memory_that_changes.union(np.where((state1 - state0) != 0)[0])
    return memory_that_changes
        

## Incremental training
# create and initialize Q (state-action value) function
Q_random = LookupTable()
X = []
for player in range(2,22):
    for dealer in range(1,22):
        for ace in range(2):
            for action in range(2):
                X.append([player, dealer, ace, action])
X = np.array(X)
Q_random.fit(X, np.random.random(X.shape[0]))
print('Mean reward per game for random agent (Q before training): %f' % np.mean([play(env,Q_random) for i in range(100000)]))
Q = train_Q_incrementally(Q_random)
print('Mean reward for game with trained agent: %f' % np.mean([play(env, Q) for i in range(100000)]))
print('Mean reward for play with Q_ideal (Vegas blackjack strategy card): %f' % np.mean([play(env, Q_ideal) for i in range(100000)]))
# Fit an SVM regression model to the lookup table version of Q to see what's
# the best we can do with an SVM
Q_SVM = sklearn.svm.SVR()
Q_SVM.fit(X, Q.predict(X))
print('Mean reward for best-fit SVM regression (SVR fit to the lookup table): %f' % np.mean([play(env, Q_SVM) for i in range(100000)]))

    
## Batch training
# Batch training using SVR regression
Q_SVR = sklearn.svm.SVR()
Q_SVR.fit(np.random.random((1000, 4)), np.random.random(1000))
Q_SVR = train_Q_batch(Q_SVR)
print('Mean reward for batch-trained SVR: %f' % np.mean([play(env, Q) for i in range(100000)]))
