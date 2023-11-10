import argparse
import torch
from datetime import datetime
from agents.ppo_discrete import PPO_discrete
from models import BC_MODELS
import torch.nn as nn
from bc.bc_hh import BehaviorClone
from utils import seed_everything, LinearAnnealer, init_env, ReplayBuffer, Normalization, RewardScaling
import wandb


def train(args, ego_agent:PPO_discrete, alt_agent:nn.Module, n_episodes:int, seed:int, logger):
    annealer = LinearAnnealer(horizon=args.num_episodes * args.max_episode_steps * 0.5)
    env = init_env(layout=args.layout, lossless_state_encoding=False)
    # print("state_dim={}".format(args.state_dim))
    # print("action_dim={}".format(args.action_dim))
    ego_buffer = ReplayBuffer(args.batch_size, args.state_dim)
    cur_steps = 0  # Record the total steps during the training
    if args.use_state_norm:
        ego_state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    for k in range(1, n_episodes+1):
        agent_env_steps = args.max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)
        obs  = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        if args.use_state_norm:
            ego_obs = ego_state_norm(ego_obs)
        
        if args.use_reward_scaling:
            reward_scaling.reset()
        
        episode_steps = 0
        done = False
        episode_reward = 0
        while not done:
            cur_steps += 1
            episode_steps += 1
            ego_a, ego_a_logprob = ego_agent.choose_action(ego_obs)  # Action and the corresponding log probability
            alt_a = alt_agent.choose_action(alt_obs, deterministic=True)  # Action and the corresponding log probability
            obs_, sparse_reward, done, info = env.step((ego_a, alt_a))
            shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
            r = sparse_reward + shaped_r*reward_shaping_factor
            ego_obs_, alt_obs_ = obs_['both_agent_obs']
            # obs = {
            #     "both_agent_obs": both_agents_ob,
            #     "overcooked_state": next_state,
            #     "other_agent_env_idx": 1 - self.agent_idx,
            # }
            episode_reward += r
            if args.use_state_norm:
                ego_obs_ = ego_state_norm(ego_obs_)
            if args.use_reward_scaling:
                r = reward_scaling(r)
            if done:
                dw = True
            else:
                dw = False
            ego_buffer.store(ego_obs, ego_a, ego_a_logprob, r, ego_obs_, dw, done)

            ego_obs = ego_obs_
            alt_obs = alt_obs_
            if ego_buffer.count == args.batch_size:
                ego_agent.update(ego_buffer, cur_steps)
                ego_buffer.count = 0
        wandb.log({'episode': k, 'ep_reward': episode_reward})
        print(f"Ep {k}:", episode_reward)
        # logger.update(score=[episode_reward], total_steps=total_steps)
    ego_agent.save_actor(f'../models/bcp/bcp_{args.layout}-seed{seed}.pth')


def run():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--use_minibatch", type=bool, default=False, help="whether sample Minibatchs during policy updating")
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--num_episodes',  type=int, default=3000)
    args = parser.parse_args()
    
    args.max_episode_steps = 600 # Maximum number of steps per episode
    args.t_max = args.num_episodes * args.max_episode_steps
    test_env = init_env(layout=args.layout, lossless_state_encoding=False)
    args.state_dim = test_env.observation_space.shape[0]
    args.action_dim = test_env.action_space.n
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    wandb.init(project='overcooked_rl',
               group='BCP',
               name=f'bcp_ppo_{args.layout}_{formatted_now}',
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

        alt_agent = torch.load(BC_MODELS[args.layout], map_location='cpu')
        # logger = Logger(exp_name=f'SP', env_name=LAYOUT_NAME)
        train(args, ego_agent=ego_agent, alt_agent=alt_agent, n_episodes=args.num_episodes, seed=seed, logger=None)  # wanghm

if __name__ == '__main__':
    run()


