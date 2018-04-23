
from __future__ import division, print_function, absolute_import

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:17:12 2018

@author: Stephen
"""

# -*- coding: utf-8 -*-
"""
Teaching a machine to play an Atari game (Pacman by default) by implementing
a 1-step Q-learning with TFLearn, TensorFlow and OpenAI gym environment. The
algorithm is described in "Asynchronous Methods for Deep Reinforcement Learning"
paper. OpenAI's gym environment is used here for providing the Atari game
environment for handling games logic and states. This example is originally
adapted from Corey Lynch's repo (url below).
Requirements:
    - gym environment (pip install gym)
    - gym Atari environment (pip install gym[atari])
References:
    - Asynchronous Methods for Deep Reinforcement Learning. Mnih et al, 2015.
Links:
    - Paper: http://arxiv.org/pdf/1602.01783v1.pdf
    - OpenAI's gym: https://gym.openai.com/
    - Original Repo: https://github.com/coreylynch/async-rl
"""
import sklearn.svm
import threading
import random
import numpy as np
import time
import os, shutil, sys
from collections import deque

import gym
import tensorflow as tf
import tflearn

global test1
global test2
global test3
global test4
global test5
# Fix for TF 0.12
try:
    writer_summary = tf.summary.FileWriter
    merge_all_summaries = tf.summary.merge_all
    histogram_summary = tf.summary.histogram
    scalar_summary = tf.summary.scalar
except Exception:
    writer_summary = tf.train.SummaryWriter
    merge_all_summaries = tf.merge_all_summaries
    histogram_summary = tf.histogram_summary
    scalar_summary = tf.scalar_summary

# Change that value to test instead of train
testing = False
# Give option to continue previous training sessions
continue_prev_train = False
# Model path (to load when testing)
test_model_path = './Pacman_Ex_files/qlearning.ckpt'
# Atari game to learn
game = 'MsPacman-ram-v0'
# Learning threads
n_threads = 32

# =============================
#   Training Parameters
# =============================
# Max training steps
TMAX = 60000000
# Current training step
T = 0
# Total number of games played
games_played = 0
# Number of frames to skip at the start of the game
start_frame = 1
# Number of screen frames to look at for training
screen_buffer_size = 4
# Number of frames between making a new decision
action_repeat = 10
# Timestep to reset the target network; 30000 frames represents 
# ~50 games in Ms Pacman.  This line and a few more contain a scaling
# factor related to "action_repeat" that controls for game lengths
I_target = np.floor(30000 / action_repeat).astype(int)
# Async gradient update frequency of each learning thread
I_AsyncUpdate = np.floor(100 / action_repeat).astype(int)
# Learning rate
learning_rate = 0.001
# Reward discount rate
gamma = 0.99
# Number of timesteps to anneal epsilon
anneal_epsilon_timesteps = np.floor(1000000 / action_repeat).astype(int)


# =============================
#   Utils Parameters
# =============================
# Display or not gym evironment screens
show_training = True
# Directory for storing tensorboard summaries
summary_dir = './tflearn_logs/'
summary_interval = np.floor(100 / action_repeat).astype(int)
checkpoint_path = './Pacman_Ex_files/qlearning.ckpt'
checkpoint_interval = np.floor(30000 / action_repeat).astype(int)
# Number of episodes to run gym evaluation
num_eval_episodes = 100


# =============================
#   TFLearn Deep Q Network
# =============================
def build_dqn(num_actions, screen_buffer_size):
    """
    Building a DQN.
    """
    inputs = tf.placeholder(tf.float32, [None, screen_buffer_size, 128, 8])
    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tf.transpose(inputs, [0, 2, 3, 1])
    net = tflearn.fully_connected(net, np.floor(1024*screen_buffer_size/2))
    net = tflearn.activations.leaky_relu(net, alpha=0.05)
    net = tflearn.fully_connected(net, 1024)
    net = tflearn.activations.leaky_relu(net, alpha=0.05)
    q_values = tflearn.fully_connected(net, num_actions)
    return inputs, q_values


#==============================
#   Convert the integer array in the observation to a binary array
#==============================
    
def vec_bin_array(arr, m):
    """
    Arguments: 
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.float32)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("float32")

    return ret 


# =============================
#   ATARI Environment Wrapper
# =============================
class AtariEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for holding on to a memory buffer
    of size screen_buffer_size from which environment state is constructed.
    """
    def __init__(self, gym_env, screen_buffer_size):
        global action_repeat
        self.env = gym_env
        self.screen_buffer_size = screen_buffer_size

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(gym_env.action_space.n)
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()

        s_t = self.env.reset()
        #x_t = vec_bin_array(x_t, 8)
        #s_t = np.stack([x_t for i in range(self.screen_buffer_size)], axis=0)
        
        for i in range(self.screen_buffer_size-1):
            self.state_buffer.append(s_t)
            
        # skip past the initial screen
        for i in range(start_frame):
            self.step(0)
        
        return s_t

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of screen_buffer_size-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """
        r_t = 0
        for i in range(action_repeat):
            s_t1, reward, terminal, info = self.env.step(self.gym_actions[action_index])
            #x_t1 = vec_bin_array(x_t1, 8)
            r_t += reward
            
            #previous_frames = np.array(self.state_buffer)
            #s_t1 = np.empty((self.screen_buffer_size, 128, 8))
            #s_t1[:self.screen_buffer_size-1, :] = previous_frames
            #s_t1[self.screen_buffer_size-1] = x_t1#

            # Pop the oldest frame, add the current frame to the queue
            #self.state_buffer.popleft()
            #self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info

   
    

# =============================
#   1-step Q-Learning
# =============================


def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1, .05, .01])
    probabilities = np.array([0.25, 0.5, 0.25])
    
    
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


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
    # return the max actiond
    return predict.argmax(axis=1)
   
   
 



	
def actor_learner_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver,Q,history,history_lock):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    global TMAX, T, games_played

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, assign_ops, summary_op = summary_ops
    Q=Q

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env, screen_buffer_size=screen_buffer_size)

    # Initialize network gradients
    s_batch = []
    a_batch = []
    y_batch = []
  
    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))
        
    time.sleep(0.5 * thread_id)
    t = 0
    game_history = GameHistory(130)
    alpha = 0.01
    initialized = False
    max_steps=1000000
    while T < TMAX:  #until T>Tmax
        history_lock.acquire()
        history_lock.release()
        h = play_history(env, Q, epsilon, max_steps,num_actions,final_epsilon,initial_epsilon,T,t,session,summary_ops,thread_id)
        game_history.add(h)
        sample = game_history.sample(5000)
        state_action = sample[:,0:-1]
        reward = sample[:,-1]
        if(initialized == True):
            Q_predict = Q_SVR.predict(state_action)
            y_train = Q_predict + alpha*(reward - Q_predict)
        else:
            y_train = sample[:,-1]
            initialized = True
        Q.fit(state_action, y_train)
        history_lock.acquire()
        h[:,-1] = np.sum(h[:,-1])  # update all rows so the reward is the total score
        history.append(h)
        history_lock.release()		
def play_history(env, Q, epsilon, max_steps,num_actions,final_epsilon,initial_epsilon,T,t,session,summary_ops,thread_id):		
    global games_played
    summary_placeholders, assign_ops, summary_op = summary_ops
    #s_t = env.get_initial_state() #receive new state
    terminal = False
    # Set up per-episode counters
    ep_reward = 0
    episode_ave_max_q = 0
    ep_t = 0
    play_history
    history = []
    s = env.get_initial_state()
    r = 0.
    d = False
    count = 0
    while True:
		
            # Forward the deep q network, get Q(s,a) values
        #readout_t = q_values.eval(session=session, feed_dict={s: [s_t]}) ## REPLACE

            # Choose next action based on e-greedy policy
			#MC
			
        if(np.random.random() < epsilon):
            action = random.randrange(num_actions)
        else:
            action = best_action(Q, np.reshape(s, (1,s.shape[0])),1)[0]
        s1, r , terminal , info = env.step(action)
        try:
            history.append(np.concatenate([s1, [action], [r]]))
        except Exception as e:
            print('got here')
            print('s1 shape: %s' % (s1.shape,))
            global test1
            global test2
            global test3
            global test4
            test1=history
            test2=s1
            test3=r
            test4=action
        s = s1
        s_t=s
        r_t=r
			
			
			
			
			
            #a_t = np.zeros([num_actions])
            #if random.random() <= epsilon:
            #    action_index = random.randrange(num_actions)
            #else:
            #    action_index = np.argmax(readout_t)
            #a_t[action_index] = 1

            # Scale down epsilon
        if epsilon > final_epsilon:
            epsilon -= (initial_epsilon - final_epsilon) / (anneal_epsilon_timesteps / n_threads)

            ## Gym excecutes action in game environment on behalf of actor-learner
            #s_t1, r_t, terminal, info = env.step(action_index)
            ## Accumulate gradients
            #readout_j1 = target_q_values.eval(session = session, feed_dict = {st : [s_t1]})###save lessons from move into a target REPLACE
            #clipped_r_t = np.clip(r_t, -1, 1)
            #if terminal:
            #    y_batch.append(clipped_r_t)
            #else:
            #    y_batch.append(clipped_r_t + gamma * np.max(readout_j1))

            #a_batch.append(a_t)
            #s_batch.append(s_t)

            ## Update the state and counters
            #s_t = s_t1
        T += 1
        t += 1

        ep_t += 1
        ep_reward += r_t
            #episode_ave_max_q += np.max(readout_t)

            # Optionally update target network
            #if T % I_target == 0:  #If T mod Itarget ==0
            #    session.run(reset_target_network_params) #Update Target Network theta- <- theta

            ## Save model progress
        if T % checkpoint_interval == 0:
            saver.save(session, "./Pacman_Ex_files/qlearning.ckpt", global_step=T)

            # Optionally update online network
            #if t % I_AsyncUpdate == 0 or terminal:
            #    if s_batch:				#Perform asynchronous update of theta using dtheta
            #        session.run(grad_update, feed_dict={y: y_batch,
            #                                            a: a_batch,
            #                                            s: s_batch})
                # Clear gradients
            #    s_batch = []	#clear dtheta
            #    a_batch = []
            #    y_batch = []

            # Print end of episode stats
        if terminal:
            games_played += 1
            stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
            for i in range(len(stats)):
                session.run(assign_ops[i],
                            {summary_placeholders[i]: float(stats[i])})
            print("| Thread %.2i" % int(thread_id), "| Step  %.2i" % int(t), 
                  "  Global Step  %.2i" % int(T), 
                  "| Reward: %.2i" % int(ep_reward), " Qmax: %.4f" %
                  (episode_ave_max_q/float(ep_t)),
                  " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                  (T/float(anneal_epsilon_timesteps)), 
                  "  Games Completed:  %.2i" % int(games_played))
            print(ep_reward)
            break
    return np.array(history)

def build_graph(num_actions):
    # Create shared deep q network
    s, q_network = build_dqn(num_actions=num_actions, screen_buffer_size=screen_buffer_size)
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network
    st, target_q_network = build_dqn(num_actions=num_actions, screen_buffer_size=screen_buffer_size)
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network

    # Op for periodically updating target network with online network weights
    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s": s,
                 "q_values": q_values,
                 "st": st,
                 "target_q_values": target_q_values,
                 "reset_target_network_params": reset_target_network_params,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}

    return graph_ops


# Set up some episode summary ops to visualize on tensorboard.
def build_summaries():
    episode_reward = tf.Variable(0.)
    scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    scalar_summary("Qmax_Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    scalar_summary("Epsilon", logged_epsilon)
    # Threads shouldn't modify the main graph, so we use placeholders
    # to assign the value of every summary (instead of using assign method
    # in every thread, that would keep creating new ops in the graph)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float")
                            for i in range(len(summary_vars))]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i])
                  for i in range(len(summary_vars))]
    summary_op = merge_all_summaries()
    
    return summary_placeholders, assign_ops, summary_op


def get_num_actions():
    """
    Returns the number of possible actions for the given atari game
    """
    
    # Figure out number of actions from gym env
    env = gym.make(game)
    num_actions = env.action_space.n

    return num_actions


def train(session, graph_ops, num_actions, saver,Q):
    """
    Train a model.
    """
    history = []
    threads = []
    history_lock = threading.Lock()
	
    global test_model_path
    
    if continue_prev_train:
        saver.restore(session, test_model_path)
        print("Restored model weights from ", test_model_path)

    # Set up game environments (one per thread)
    envs = [gym.make(game) for i in range(n_threads)]

    summary_ops = build_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.global_variables_initializer())
    writer = writer_summary(summary_dir + "./qlearning/", session.graph)

    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    # Start "n_ts" actor-learner training threads
    actor_learner_threads = \
        [threading.Thread(target=actor_learner_thread,
                          args=(thread_id, envs[thread_id], session,
                                graph_ops, num_actions, summary_ops, saver,Q,history,history_lock))
         for thread_id in range(n_threads)]
    for t in actor_learner_threads:
        t.start()
        time.sleep(0.01)

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
#        if show_training:
#            index_env = 0
#            for env in envs:
#                if index_env == 0:
#                    env.render()
#                    index_env = 1
        now = time.time()
        if now - last_summary_time > summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()
    history = np.concatenate(history)
    return history


def evaluation(session, graph_ops, saver):
    """
    Evaluate a model.
    """
    saver.restore(session, test_model_path)
    print("Restored model weights from ", test_model_path)
    monitor_env = gym.make(game)
#    monitor_env = gym.wrappers.Monitor(monitor_env, "./qlearning/eval")

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env,
                           screen_buffer_size=screen_buffer_size)

    for i_episode in range(num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
#            monitor_env.render()
            readout_t = q_values.eval(session=session, feed_dict={s : [s_t]})
            action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
    monitor_env.close()


def main(_):
    Q=sklearn.svm.SVR(kernel='rbf')
    with tf.Session() as session:
        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        saver = tf.train.Saver(max_to_keep=np.ceil(TMAX / checkpoint_interval).astype(int))

        if testing:
            evaluation(session, graph_ops, saver)
        else:
            train(session, graph_ops, num_actions, saver,Q)

if __name__ == "__main__":
    tf.app.run()
    
