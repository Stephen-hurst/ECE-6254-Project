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
from __future__ import division, print_function, absolute_import

import threading
import random
import numpy as np
import time
import os, shutil, sys
from collections import deque

import gym
import tensorflow as tf
import tflearn

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
# Model path (to load when testing)
test_model_path = './Pacman_Ex_files/qlearning.ckpt-718000'
# Atari game to learn
# You can also try: 'Breakout-v0', 'Pong-v0', 'SpaceInvaders-v0', ...
game = 'MsPacman-ram-v0'
# Learning threads
n_threads = 8

# =============================
#   Training Parameters
# =============================
# Max training steps
TMAX = 8000000
# Current training step
T = 0
# Total number of games played and training start time
games_played = 0
start_time = time.time()
# Consecutive screen frames when performing training
action_repeat = 2
# Reduce score per timestep to motivate toward completion
time_cost = 0.2
# Async gradient update frequency of each learning thread
I_AsyncUpdate = 100
# Timestep to reset the target network
I_target = 37500
# Learning rate
learning_rate = 0.001
# Reward discount rate
gamma = 0.99
# Number of timesteps to anneal epsilon
anneal_epsilon_timesteps = 500000

# =============================
#   Utils Parameters
# =============================
# Display or not gym evironment screens
show_training = True
# Directory for storing tensorboard summaries
summary_dir = './tflearn_logs/'
summary_interval = 100
checkpoint_path = './Pacman_Ex_files/qlearning.ckpt'
checkpoint_interval = 2000
# Number of episodes to run gym evaluation
num_eval_episodes = 100


# =============================
#   TFLearn Deep Q Network
# =============================
def build_dqn(num_actions, action_repeat):
    """
    Building a DQN.
    """
    inputs = tf.placeholder(tf.float32, [None, action_repeat, 128, 8])
    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tf.transpose(inputs, [0, 2, 3, 1])
    net = tflearn.fully_connected(net, 1024)
    net = tflearn.activations.leaky_relu(net, alpha=0.1)
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
    of size action_repeat from which environment state is constructed.
    """
    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.action_repeat = action_repeat

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(0,5)  # range(gym_env.action_space.n)
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = vec_bin_array(x_t, 8)
        s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)
        
        for i in range(self.action_repeat-1):
            self.state_buffer.append(x_t)
        return s_t

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = vec_bin_array(x_t1, 8)
        r_t = self.reward_adjust(r_t)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.action_repeat, 128, 8))
        s_t1[:self.action_repeat-1, :] = previous_frames
        s_t1[self.action_repeat-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info

    def reward_adjust(self, reward):
        if not testing:
            # adjust for wasting time; encourages pacman to move through maze
            reward -= time_cost
            
        return reward
    
    

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


def actor_learner_thread(thread_id, env, session, graph_ops, num_actions,
                         summary_ops, saver):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    global TMAX, T

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

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env,
                           action_repeat=action_repeat)

    # Initialize network gradients
    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))

    time.sleep(3*thread_id)
    t = 0
    while T < TMAX:
        # Get initial game observation
        s_t = env.get_initial_state()
        terminal = False

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        while True:
            # Forward the deep q network, get Q(s,a) values
            readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})

            # Choose next action based on e-greedy policy
            a_t = np.zeros([num_actions])
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            # Scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps

            # Gym excecutes action in game environment on behalf of actor-learner
            s_t1, r_t, terminal, info = env.step(action_index)
            # Accumulate gradients
            readout_j1 = target_q_values.eval(session = session,
                                              feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + gamma * np.max(readout_j1))

            a_batch.append(a_t)
            s_batch.append(s_t)

            # Update the state and counters
            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # Optionally update target network
            if T % I_target == 0:
                session.run(reset_target_network_params)

            # Optionally update online network
            if t % I_AsyncUpdate == 0 or terminal:
                if s_batch:
                    session.run(grad_update, feed_dict={y: y_batch,
                                                        a: a_batch,
                                                        s: s_batch})
                # Clear gradients
                s_batch = []
                a_batch = []
                y_batch = []

            # Save model progress
            if t % checkpoint_interval == 0:
                saver.save(session, "./Pacman_Ex_files/qlearning.ckpt", global_step=t)

            # Print end of episode stats
            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(assign_ops[i],
                                {summary_placeholders[i]: float(stats[i])})
                print("| Thread %.2i" % int(thread_id), "| Step", t,
                      "| Reward: %.2i" % int(ep_reward), " Qmax: %.4f" %
                      (episode_ave_max_q/float(ep_t)),
                      " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                      (t/float(anneal_epsilon_timesteps)))
                break


def build_graph(num_actions):
    # Create shared deep q network
    s, q_network = build_dqn(num_actions=num_actions,
                             action_repeat=action_repeat)
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network
    st, target_q_network = build_dqn(num_actions=num_actions,
                                     action_repeat=action_repeat)
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
    scalar_summary("Qmax Value", episode_ave_max_q)
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
    games_played = len(summary_vars)
    
    return summary_placeholders, assign_ops, summary_op


def get_num_actions():
    """
    Returns the number of possible actions for the given atari game
    """
    
    """
    # Figure out number of actions from gym env
    env = gym.make(game)
    num_actions = env.action_space.n
    """
    
    # set to 4 for our purposes to reduce computational complexity and force
    # an action (as opposed to NOP)
    num_actions = 5
    return num_actions


def train(session, graph_ops, num_actions, saver):
    """
    Train a model.
    """
    
#    if os.listdir("./Pacman_Ex_files/")!="":
#        saver.restore(session, test_model_path)

    # Set up game environments (one per thread)
    envs = [gym.make(game) for i in range(n_threads)]

    summary_ops = build_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.global_variables_initializer())
    writer = writer_summary(summary_dir + "/qlearning", session.graph)

    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    # Start "n_threads" actor-learner training threads
    actor_learner_threads = \
        [threading.Thread(target=actor_learner_thread,
                          args=(thread_id, envs[thread_id], session,
                                graph_ops, num_actions, summary_ops, saver))
         for thread_id in range(n_threads)]
    for t in actor_learner_threads:
        t.start()
        time.sleep(0.01)

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        if show_training:
            index_env = 0
            for env in envs:
                if index_env == 0:
                    env.render()
                    index_env = 1
        now = time.time()
        if now - last_summary_time > summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()


def evaluation(session, graph_ops, saver):
    """
    Evaluate a model.
    """
    saver.restore(session, test_model_path)
    print("Restored model weights from ", test_model_path)
    monitor_env = gym.make(game)
    monitor_env = gym.wrappers.Monitor(monitor_env, "./qlearning/eval")

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env,
                           action_repeat=action_repeat)

    for i_episode in range(num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session=session, feed_dict={s : [s_t]})
            action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print(ep_reward)
    monitor_env.close()


def main(_):
    with tf.Session() as session:
        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        saver = tf.train.Saver(max_to_keep=5)

        if testing:
            evaluation(session, graph_ops, saver)
        else:
            train(session, graph_ops, num_actions, saver)

if __name__ == "__main__":
    tf.app.run()
    
