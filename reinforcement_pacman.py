# ECE6254 Project - Reinforcement Learning to play Ms Pacman
import gym
import tensorflow as tf
import numpy as np
import sklearn.svm
import sys
import threading
sys.path.append('ECE-6254-Project')
from lookup_table import LookupTable
from game_history import GameHistory

## Set Parameters
alpha = 0.01  # determines how fast we update Q, the state-value table
y = 0.99  # gamma: discount on future rewards

## Initialize environment and variables
# create OpenAI gym environment
env = gym.make('MsPacman-ram-v0')
# only use memory locations that change through the game (predetermined by
# calling find_changing_memory_locations)
memory_that_changes = np.arange(128)
OBSERVATIONS = memory_that_changes.shape[0]
ACTIONS = 9  # simplify actions to up, down (==> 2, 5 for Atari Learning Env.)
MAX_SAMPLES = 6000  # maximum number of state, action, reward, state' samples to store
GAMES_PER_EPISODE = 3
NUM_EPISODES = 200000  # number of episodes (e.g., batches of games; anytime we retrain our policy model) we should do
SAMPLES_TO_TRAIN_ON = 6000  # how many samples from our replay memory should we train Q (our policy) on?
eps = np.log(0.0001)/NUM_EPISODES  # chance of random exploration vs. choosing best policy (used in np.exp)


def normalize_state(state):
    '''Normalize a state directly from the OpenAI gym environment, i.e.
    pick out only useful features, subtract the mean, and divide by variance'''
    state = state[memory_that_changes]
    state = np.array(state, dtype='float64')
    state -= 128.
    state /= 256.
    return state
    
    
def normalize_reward(reward):
    '''Normalize a reward directly from OpenAI gym environment'''
    return reward/10.


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
    

def play(env, Q, render=False, max_points=100000):
    '''Play a game using policy Q and return the reward'''
    s = env.reset()
    s = normalize_state(s)
    s1 = s
    reward = 0
    total_reward = 0
    total_points = 0
    done = False
    while(done is False and total_points <= max_points):
        if(type(Q) == np.ndarray):
            action = Q[tuple_to_int(s),:].argmax()
        else:
            action = best_action(Q, s.reshape((1, s.shape[0])))[0]
        s1, reward, done, _ = env.step(action)
        if(render == True):
            env.render()
        s1 = normalize_state(s1)
        total_reward += reward
        total_points += np.round(np.abs(reward))
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
    num_batches = NUM_EPISODES
    replay_memory = np.zeros((0, OBSERVATIONS*2+3))  # history of (s, a, r, s') experiences
    for batch_i in range(num_batches):
        print('Batch %d' % batch_i)
        for game_i in range(GAMES_PER_EPISODE):
            '''Play a single game with epsilon-greedy planning and update the replay memory'''
            s = env.reset()
            s = normalize_state(s)
            r = 0.0
            total_reward = 0.0
            done = False
            while(done is False):
                # create a matrix with the current state and all actions
                #action = best_action(Q, s.reshape((1, s.shape[0])))[0]
                action = env.action_space.sample()
                # use a decaying random exploration rate to sometimes try
                # random actions
                if(np.random.random() < np.exp(batch_i*eps)):
                    action = env.action_space.sample()
                s1, r, done, _ = env.step(action)
                s1 = normalize_state(s1)
                r = normalize_reward(r)
                if(done == True):
                    r += total_reward  # add the total score as a big reward if this game is over
                    print('Score: %d' % (total_reward*10))
                this_experience = np.array([np.concatenate([s1, [action], [r], [done], s1])])
                if(replay_memory.shape[0] >= MAX_SAMPLES):
                    replay_memory[np.random.randint(0, MAX_SAMPLES-1),:] = this_experience
                else:
                    replay_memory = np.concatenate([replay_memory, this_experience])
                s = s1
                total_reward += r
        # Update Q using the current replay memory
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
    
    memory_that_changes = set()
    
    for game_i in range(games_to_play):
        state0 = env.reset()
        reward = 0.
        done = False
        while(done == False):
            state1, reward, done, _ = env.step(env.action_space.sample())
            memory_that_changes = memory_that_changes.union(np.where((state1 - state0) != 0)[0])
    return memory_that_changes
        

## Batch training
# Batch training using SVR regression
Q_SVR = sklearn.svm.SVR()
Q_SVR.fit(np.random.random((1000, OBSERVATIONS+1)), np.random.random(1000))
Q_SVR = train_Q_batch(Q_SVR)
#print('Mean reward in Pong for batch-trained SVR: %f' % np.mean([play(env, Q) for i in range(100000)]))
