import argparse
import time
import gym
from monitor import TimeLimitMonitor
import numpy as np
import itertools
import torch
import os
import sys
from sac import SAC
from tensorboardX import SummaryWriter
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
from eval import policy_evaluation, MyLogger


USE_CUDA = torch.cuda.is_available()
def set_device(gpu: int):
    if gpu >= 0 and USE_CUDA:
        torch.cuda.set_device(gpu)

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--log_dir', default="", type=str,
                    help='Log dir')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU ID')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=2018, metavar='N',
                    help='random seed (default: 2018)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()
set_device(args.gpu)

video_base = os.path.join(args.log_dir, 'videos')
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(video_base, exist_ok=True)


# Environment
env = NormalizedActions(TimeLimitMonitor(gym.make(args.env_name), video_base, True, video_callable=lambda x: x % 100 == 0))

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

writer = SummaryWriter(args.log_dir)
eval_logger = MyLogger(args.log_dir, 'po_eval')
# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
rewards = []
test_rewards = []
total_numsteps = 0
updates = 0

video_recorder = None

for i_episode in itertools.count(1):
    state = env.reset()
    episode_reward = 0

    while True:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)  # Sample action from policy
        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step): # Number of updates per step in environment
                # Sample a batch from memory
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.batch_size)
                # Update parameters of all the networks
                value_loss, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(state_batch, action_batch,
                                                                                                reward_batch, next_state_batch, 
                                                                                                mask_batch, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        mask = float(not done)  # 1 for not done and 0 for done

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        total_numsteps += 1
        episode_reward += reward

        sys.stdout.write(f"\rStep {total_numsteps}, updates {updates}")
        sys.stdout.flush()

        if total_numsteps % 1000 == 0:
            eval_result = policy_evaluation(agent, args.env_name)
            print("\n----------------------------------------")
            print("Test Episode: {}, reward: {}".format(total_numsteps, eval_result))
            print("----------------------------------------")
            writer.add_scalar("Episode Return/Evaluation", eval_result, total_numsteps)
            eval_logger.write(total_numsteps, eval_result)

        if done:
            break

    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    writer.add_scalar("Episode Return/Current episode", episode_reward, total_numsteps)
    rewards.append(episode_reward)
    print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}\n".format(i_episode, total_numsteps, np.round(rewards[-1],2),
                                                                                np.round(np.mean(rewards[-100:]),2)))
env.close()


