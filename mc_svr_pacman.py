import gym
import numpy as np
import sklearn.svm
#import matplotlib.pyplot as plt

env = gym.make('MsPacman-ram-v0')
import threading
THREADS = 8

class GameHistory():
    '''Maintain a history of state-action-value tuples, replacing randomly
    when we exceed the max storage'''
    def __init__(self, features, max_size=50000):
        self.max_size = max_size
        self._lock = threading.Lock()
        self._storage = np.zeros((0,features))
        
    def add(self, samples):
        '''Add N rows (samples is a matrix with N rows and some features)'''
        self._lock.acquire()
        try:
            if(self._storage.shape[0] < self.max_size):
                self._storage = np.concatenate([self._storage, samples])
                # chop if we went over
                if(self._storage.shape[0] > self.max_size):
                    self._storage = self._storage[0:self.max_size,:]
            else:
                rows_to_replace = np.random.choice(self.max_size, samples.shape[0])
                self._storage[rows_to_replace,:] = samples
        except Exception as e:
            raise e
        finally:
            self._lock.release()
        
    def sample(self, N):
        '''Return N random rows'''
        rows = np.random.choice(self._storage.shape[0], N)
        return self._storage[rows,:]
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

def train_Q_incrementally(Q, plot_every=None):
    # Train Q, the state-value table by playing a bunch of games and updating
    # Q via the Bellman equation
    # Q function is implemented as an arbitrary model with .fit and .predict
    # functions
    OBSERVATIONS = 3
    ACTIONS = 2
    SAMPLES_TO_TRAIN_ON = 1
    mean_scores = []  # if plot_every is not None, track and plot performance during training 
    for e_i in range(num_episodes):
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
                Q.fit(S_A, Q_S_A + alpha*(reward + y*best_action_value(Q, S_1, ACTIONS) - Q_S_A))
            player, dealer, ace = s1
        if(plot_every != None and (e_i % plot_every) == 0):
            mean_scores.append(np.mean([play(env, Q) for i in range(10000)]))
    if(plot_every != None):
        plt.plot(np.arange(0, len(mean_scores)*plot_every, plot_every), np.array(mean_scores))
    return Q
    
def train_Q_batch(Q, alpha=1.0):
    '''Train Q using a batch method. Q should already be initialized so
    Q.predict works.'''
    # Train using a batch of several games at once
    # Use experience replay and fixed-Q targets to update Q
    # experience replay: keep a replay history (s, a, r, s') and train using
    # samples from that history
    # fixed-Q: use a previous version of Q in each batch run
    OBSERVATIONS = 3
    ACTIONS = 2
    SAMPLES_TO_TRAIN_ON = 1000
    batch_size = 100  # how many games do we play per batch
    num_batches = 20
    replay_memory = np.zeros((0, OBSERVATIONS*2+3))  # history of (s, a, r, s') experiences
    for batch_i in range(num_batches):
        print('Batch %d' % batch_i)
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
                if(np.random.random() < np.exp(batch_i/num_batches*eps)):
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
        Q_predict = Q.predict(S_A).reshape(S_A.shape[0],)
        if('train_on_batch' in dir(Q)):
            y_train = reward_column + y*max_a_s1
            Q.train_on_batch(S_A, y_train)
        else:
            y_train = Q_predict + alpha*(reward_column + y*max_a_s1 - Q_predict)
            Q.fit(S_A, y_train)
    return Q

def play_all_games(num_games, Q, eps=1.0, max_steps=100000):
    '''Play a whole bunch of games and return the state, action, reward
    history as a matrix. Uses threads to speed things up.'''
    history = []
    threads = []
    history_lock = threading.Lock()
    # Run a certain number of games and append to the history
    # this is the function we'll run in threads
    def play_games_in_thread(thread_id, games_per_thread, history, history_lock):
        for g_i in range(games_per_thread):
            print('Thread %d game %d' % (thread_id, g_i))
            history_lock.acquire()
            env = gym.make('MsPacman-ram-v0')
            history_lock.release()
            h = play_history(env, Q, eps, max_steps)
            history_lock.acquire()
            h[:,-1] = np.sum(h[:,-1])  # update all rows so the reward is the total score
            history.append(h)    
            env.close()
            history_lock.release()
    # Start up threads
    for t_i in range(THREADS):
        t = threading.Thread(target=play_games_in_thread, args=[t_i, int(num_games/THREADS), history, history_lock])
        threads.append(t)
        t.start()
    # wait for threads to finish
    for t in threads:
        t.join()
    history = np.concatenate(history)
    return history
    

def play_history(env, Q, eps=1.0, render=False, max_steps=100):
    '''Play a game with an epsilon-greedy policy and return a history of states, actions, and rewards'''
    history = []
    s = env.reset()
    # the game doesn't do anything for the first several time steps
    for i in range(89):
        env.step(0)
    r = 0.
    d = False
    count = 0
    while(d == False and count < max_steps):
        if(np.random.random() < eps):
            action = env.action_space.sample()
        else:
            action = best_action(Q, np.reshape(s, (1,s.shape[0])),1)[0]
        s1, r, d, _ = env.step(action)
        try:
            history.append(np.concatenate([s1, [action], [r]]))
        except Exception as e:
            print('got here')
            print('s1 shape: %s' % s1.shape)
            raise e
        s = s1
        if(render == True):
            env.render()
        count += 1
    return np.array(history)
    
# Play a bunch of Monte-Carlo batches with 8 games (~5600 action-steps) each
# Play to a max of 100 steps to figure out the best early-game strategy
# see if we increase the average game score
MAX_STEPS = 100
game_history = GameHistory(130)
Q_SVR = sklearn.svm.SVR(kernel='rbf')
alpha = 0.01  # how fast we update Q, the action-value function
final_eps = 0.01  # final probability of a random action at the end of our training
initialized = False
mean_game_scores = []
for e_i in range(5000):
    eps = np.exp(np.log(final_eps)/1000.*e_i)
    h = play_all_games(THREADS, Q_SVR, eps, max_steps=MAX_STEPS)
    game_history.add(h)
    # use the Bellman equation to update the Q(s,a) predictions
    sample = game_history.sample(5000)
    state_action = sample[:,0:-1]
    reward = sample[:,-1]
    if(initialized == True):
        Q_predict = Q_SVR.predict(state_action)
        y_train = Q_predict + alpha*(reward - Q_predict)
    else:
        y_train = sample[:,-1]
        initialized = True
    Q_SVR.fit(state_action, y_train)
    print('Iteration %d SVR fit score: %f' % (e_i, Q_SVR.score(state_action, reward)))
    print('Iteration %d Mean game score: %f' % (e_i, np.mean(reward)))
    mean_game_scores.append(np.mean(reward))

# Save the mean game scores
np.savez('mean_game_scores-%d.npz' % MAX_STEPS, mean_game_scores)

# Save the Q_SVR fit parameters so we can retrain later
np.savez('Q_SVR_last_fit-%d.npz' % MAX_STEPS, state_action, y_train)

# # Plot the convergence (shown by mean game scores)
# plt.plot(mean_game_scores)
# plt.title('Mean game scores')
# plt.xlabel('Training episodes (8 games per episode')
# plt.ylabel('Mean game score over 8 games')
# plt.show()
