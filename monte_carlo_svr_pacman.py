import gym
import numpy as np
import sklearn.svm
#import matplotlib.pyplot as plt

env = gym.make('MsPacman-ram-v0')
import threading
THREADS = 8


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
            action = best_action(Q, np.reshape(s, (1,s.shape[0])))[0]
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
MAX_STEPS = 500
game_history = GameHistory(130)
Q_SVR = sklearn.svm.SVR(kernel='rbf')
alpha = 0.01  # how fast we update Q, the action-value function
final_eps = 0.01  # final probability of a random action at the end of our training
initialized = False
mean_game_scores = []
for e_i in range(1000):
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
