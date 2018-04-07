'''Play a random game of MsPacman'''
import gym

env = gym.make('MsPacman-ram-v0')

s = env.reset()
reward = 0
done = False
total_reward = 0

while(done is False):
    action = env.action_space.sample()
    s1, reward, done, _ = env.step(action)
    env.render()
    total_reward += reward
    s = s1

env.close()