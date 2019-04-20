import gym
import numpy as np
import sys
import os


class MyLogger(object):
    def __init__(self, log_dir, log_name):
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, f"{log_name}.npz")
        self.data = np.array([])

    def write(self, step, value):
        self.data = np.append(self.data, np.array([step, value]))
        np.savez(self.log_path, **{self.log_name: self.data})


def policy_evaluation(agent, env_id, episode=100, random_seed=1000):
    env = gym.make(env_id)
    env.seed(random_seed)
    reward_history = []
    print("\nEvaluating...")
    for i in range(episode):
        state = env.reset()
        done = False
        episode_return = 0
        time_step = 0
        while not done:
            time_step += 1
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_return += reward
            sys.stdout.write(f"\rEvaluating episode {i}, at step {time_step}, eval rewards={episode_return}")
            sys.stdout.flush()
        reward_history.append(episode_return)
    return np.mean(reward_history)
