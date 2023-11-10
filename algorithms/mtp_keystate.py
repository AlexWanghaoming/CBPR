import argparse
import torch
from datetime import datetime
from agents.ppo_discrete import PPO_discrete
from models import BC_MODELS, META_TASK_MODELS
import torch.nn as nn
from bc.bc_hh import BehaviorClone
from utils import seed_everything, LinearAnnealer, init_env, ReplayBuffer, Normalization, RewardScaling
from rl_plotter.logger import Logger
import wandb


def train(args, ego_agent:PPO_discrete, alt_agent:nn.Module, n_episodes:int, seed:int, logger=None):
    annealer = LinearAnnealer(horizon=args.num_episodes * args.max_episode_steps * 0.5)
    env = init_env(layout=args.layout, lossless_state_encoding=False)
    ego_buffer = ReplayBuffer(args.batch_size, args.state_dim)  # 主模型的buffer
    ego_reward_scaling = RewardScaling(shape=1, gamma=args.gamma) # 主模型的reward_scaling

    cur_steps = 0  # Record the total steps during the training
    meta_task_buffer = {}
    mtp_agents = {}
    meta_task_steps = {}
    meta_task_state_norm = {}
    meta_task_reward_scaling = {}
    for k in range(1, n_episodes+1):
        agent_env_steps = args.max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)
        obs = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        ego_reward_scaling.reset()

        key = tuple(alt_obs[slice(4, 8)])  # 提取key state identifier
        if key not in meta_task_state_norm:
            meta_task_reward_scaling[key] = RewardScaling(shape=1, gamma=args.gamma)

        meta_task_reward_scaling[key].reset()
        
        episode_steps = 0
        done = False
        episode_reward = 0
        while not done:
            cur_steps += 1
            episode_steps += 1
            key = tuple(alt_obs[slice(4, 8)])  # 提取key state identifier
            if key not in meta_task_buffer:
                meta_task_buffer[key] = ReplayBuffer(args.batch_size, args.state_dim)
                mtp_agents[key] = PPO_discrete(lr=args.lr,
                             hidden_dim=args.hidden_dim,
                             batch_size=args.batch_size,
                            use_minibatch=args.use_minibatch,
                            mini_batch_size=args.mini_batch_size,
                             epsilon=args.epsilon,
                             entropy_coef=args.entropy_coef,
                             state_dim=args.state_dim,
                             action_dim=args.action_dim,
                             state_value_tau=args.state_value_tau,
                             num_episodes=args.num_episodes,
                             device=args.device)
                meta_task_steps[key] = 0
                meta_task_reward_scaling[key] = RewardScaling(shape=1, gamma=args.gamma)

            meta_task_steps[key] += 1

            # ego_a, ego_a_logprob = mtp_agents[key].choose_action(ego_obs) # 用mtp模型选动作
            ego_a, ego_a_logprob = ego_agent.choose_action(ego_obs)  # 用主模型选动作
            alt_a = alt_agent.choose_action(alt_obs, deterministic=True)  # 人类模型选择的动作

            obs_, sparse_reward, done, info = env.step((ego_a, alt_a))
            shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
            r = sparse_reward + shaped_r*reward_shaping_factor
            episode_reward += r
            ego_obs_, alt_obs_ = obs_['both_agent_obs']
            r = ego_reward_scaling(r)
            m_r = meta_task_reward_scaling[key](r)

            if done:
                dw = True
            else:
                dw = False

            ego_buffer.store(ego_obs, ego_a, ego_a_logprob, r, ego_obs_, dw, done)  # 主模型的buffer存储状态
            meta_task_buffer[key].store(ego_obs, ego_a, ego_a_logprob, m_r, ego_obs_, dw, done)  # MTP模型的buffer存储状态
            ego_obs = ego_obs_
            alt_obs = alt_obs_
            ## 网络参数更新
            if ego_buffer.count == args.batch_size:
                ego_agent.update(ego_buffer, cur_steps)  # 更新主模型网络
                ego_buffer.count = 0
            if meta_task_buffer[key].count == args.batch_size:
                mtp_agents[key].update(meta_task_buffer[key], meta_task_steps[key])   # 更新MTP模型网络
                meta_task_buffer[key].count = 0

        print(f"Ep {k}:", episode_reward)
        wandb.log({'episode':k,'ep_reward': episode_reward})
        # logger.update(score=[episode_reward], total_steps=cur_steps)
    # ego_agent.save_actor(f'../models/bcp/bcp_{args.layout}-seed{seed}.pth')  # 保存主模型

    for key in mtp_agents:
        mtp_agents[key].save_actor(f'../models/mtp/mtp_{args.layout}-{key}-seed{seed}-gg.pth')


def run():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size, num of samples in buffer")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--use_minibatch", type=bool, default=False, help="whether sample Minibatchs during policy updating")
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--state_value_tau", type=float, default=0, help="the tau of normalize for value and state `std = (1-std)*std + tau*std`")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--num_episodes',  type=int, default=10000)
    args = parser.parse_args()
    args.max_episode_steps = 600 # Maximum number of steps per episode
    test_env = init_env(layout=args.layout, lossless_state_encoding=False)
    args.state_dim = test_env.observation_space.shape[0]
    args.action_dim = test_env.action_space.n
    print(args)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    wandb.init(project='overcooked_rl',
               group='MTP_keystat',
               name=f'mtp_{args.layout}_{formatted_now}',
               config=vars(args))

    # seeds = [0, 1, 42, 2022, 2023]
    seeds = [42]
    for seed in seeds:
        seed_everything(seed=seed)
        ego_agent = PPO_discrete(lr=args.lr,
                             hidden_dim=args.hidden_dim,
                             batch_size=args.batch_size,
                            use_minibatch=args.use_minibatch,
                            mini_batch_size=args.mini_batch_size,
                             epsilon=args.epsilon,
                             entropy_coef=args.entropy_coef,
                             state_dim=args.state_dim,
                             action_dim=args.action_dim,
                             num_episodes=args.num_episodes,
                             device=args.device)
        # key state mtp training
        alt_agent = torch.load(BC_MODELS[args.layout], map_location='cpu')
        train(args, ego_agent=ego_agent, alt_agent=alt_agent, n_episodes=args.num_episodes, seed=seed)

if __name__ == '__main__':
    run()


