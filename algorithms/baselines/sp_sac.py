import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../PPO-discrete/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from typing import *
# warnings.filterwarnings('ignore', message='Passing (type, 1) or \'1type\' as a synonym of type is deprecated')
from rl_plotter.logger import Logger
from My_utils import seed_everything, LinearAnnealer, init_env
import argparse
from agents.sac import SAC, OffpolicyReplayBuffer
import warnings

warnings.filterwarnings('ignore', message='Passing (type, 1) or \'1type\' as a synonym of type is deprecated')

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''train SAC agent''')
    parser.add_argument('--layout', default='cramped_room', help='layout name')
    parser.add_argument('--num_episodes',  type=int, default=1500, help='total episodes')
    args = parser.parse_args()
    return args


def trainTwoOffPolicyAgent(args,
                           env,
                           ego_agent,
                           alt_agent,
                           buffer_size,
                           reward_scale,
                           num_episodes,
                           learning_starts,
                           batch_size,
                           ):
    annealer = LinearAnnealer(horizon=args.t_max/2)

    ego_replay_buffer = OffpolicyReplayBuffer(capacity=buffer_size)
    alt_replay_buffer = OffpolicyReplayBuffer(capacity=buffer_size)
    logger = Logger(exp_name=f'sp_sac', env_name=args.layout)
    t_env = 0
    for k in range(1, num_episodes+1):
        agent_env_steps = args.max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)
        episode_reward = 0
        obs = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        done = False
        while not done:
            t_env += 1
            ego_act = ego_agent.take_action(ego_obs)
            alt_act = alt_agent.take_action(alt_obs)
            # print("actions:", (ego_act, alt_act))
            obs_, sparse_reward, done, info = env.step((ego_act, alt_act))
            shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
            r = sparse_reward + shaped_r * reward_shaping_factor

            ego_obs_, alt_obs_ = obs_['both_agent_obs']
            episode_reward += r
            r = reward_scale * r  # 对奖励进行缩放
            ego_replay_buffer.add(ego_obs, ego_act, r, ego_obs_, done)
            alt_replay_buffer.add(alt_obs, alt_act, r, alt_obs_, done)
            ego_obs, alt_obs = ego_obs_, alt_obs_
            for agent, replay_buffer in zip([ego_agent, alt_agent], [ego_replay_buffer, alt_replay_buffer]):
                if replay_buffer.size() > learning_starts:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s,
                                       'actions': b_a,
                                       'next_states': b_ns,
                                       'rewards': b_r,
                                       'dones': b_d}
                    agent.update(transition_dict, t_env)

        print(f"Ep {k} reward: {episode_reward}")
        # logger.update(score=[episode_reward], total_steps=k)
    # ego_agent.save_actor(f'{DIR}/models/policy_pool/sp_{args.layout}-seed{seed}.pth')


def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''train SAC agent''')
    parser.add_argument('--layout', default='cramped_room', help='layout name')
    parser.add_argument('--num_episodes',  type=int, default=1500, help='total episodes')
    args = parser.parse_args()

    args.max_episode_steps = 600  # Maximum number of steps per episode
    args.t_max = args.num_episodes * args.max_episode_steps

    env = init_env(layout=args.layout)
    # seeds = [0, 1, 42, 2022, 2023]
    seeds = [42]
    for seed in seeds:
        ego_agent = SAC(env, hidden_dim=64,
                        lr=1e-4,
                        tau=0.005,
                        adaptive_alpha=True,
                        clip_grad_norm=0.1,  # 0.1 ~ 4.0, clip the gradient after normalization
                        use_lr_decay=False,
                        device='cpu')

        alt_agent = SAC(env, hidden_dim=64,
                        lr=1e-4,
                        tau=0.005,
                        adaptive_alpha=True,
                        clip_grad_norm=0.1,  # 0.1 ~ 4.0, clip the gradient after normalization
                        use_lr_decay=False,
                        device='cpu')

        trainTwoOffPolicyAgent(args,
                               env,
                               ego_agent,
                               alt_agent,
                               buffer_size=1000000,
                               reward_scale = 1,
                               num_episodes=args.num_episodes,
                               learning_starts=200,
                               batch_size=256
                               )

if __name__ == '__main__':

    run()


