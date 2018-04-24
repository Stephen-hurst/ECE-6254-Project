'''Play a random game of MsPacman'''
import gym

env = gym.make('MsPacman-ram-v0')

def play_a_game(render=False):
    '''Plays a game of MsPacman and returns the final score'''
    s = env.reset()
    reward = 0
    done = False
    total_reward = 0
    
    while(done is False):
        action = env.action_space.sample()
        s1, reward, done, _ = env.step(action)
        if(render == True):
            env.render()
        total_reward += reward
        s = s1
    env.close()
    return total_reward

def game_statistics(num_games):
    '''Calculate game statistics for some number of games'''
    game_scores = [play_a_game() for i in range(num_games)]
    mean = np.mean(game_scores)
    median = np.median(game_scores)
    max = np.max(game_scores)
    print('Mean: %f Median: %f Max: %f' % (mean, median, max))
    return (mean, median, max)