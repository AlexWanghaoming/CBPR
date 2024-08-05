import argparse
import torch
from datetime import datetime
import sys, os
# print("当前系统路径", sys.path)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
# print("当前系统路径", sys.path)
from agents.ppo_discrete import PPO_discrete
from models import BC_MODELS
import torch.nn as nn
from bc.bc_hh import BehaviorClone
from My_utils import seed_everything, LinearAnnealer, init_env, ReplayBuffer, Normalization, RewardScaling
import wandb

# add = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = add
# os.environ['https_proxy'] = add
WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'


def train(args, ego_agent, alt_agent, n_episodes:int, seed:int, logger):
    annealer = LinearAnnealer(horizon=args.num_episodes * args.max_episode_steps * 0.5)
    env = init_env(layout=args.layout)
    replay_buffer = ReplayBuffer(args.batch_size, args.state_dim)
    cur_steps = 0  # Record the total steps during the training
    if args.use_state_norm:
        ego_state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    for k in range(1, n_episodes+1):
        agent_env_steps = args.max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)
        obs = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        # if args.alt:
        #     alt_obs, ego_obs = obs['both_agent_obs']

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
            if args.alt:
                alt_a, alt_a_logprob = alt_agent.choose_action(alt_obs)
                ego_a = ego_agent.choose_action(ego_obs, deterministic=True)
            else:
                ego_a, ego_a_logprob = ego_agent.choose_action(ego_obs)
                alt_a = alt_agent.choose_action(alt_obs, deterministic=True)
            obs_, sparse_reward, done, info = env.step((ego_a, alt_a))
            shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
            r = sparse_reward + shaped_r*reward_shaping_factor
            ego_obs_, alt_obs_ = obs_['both_agent_obs']
            episode_reward += r
            if args.use_state_norm:
                ego_obs_ = ego_state_norm(ego_obs_)
            if args.use_reward_scaling:
                r = reward_scaling(r)
            if done:
                dw = True
            else:
                dw = False
            if args.alt:
                replay_buffer.store(alt_obs, alt_a, alt_a_logprob, r, alt_obs_, dw, done)
            else:
                replay_buffer.store(ego_obs, ego_a, ego_a_logprob, r, ego_obs_, dw, done)
            ego_obs = ego_obs_
            alt_obs = alt_obs_
            if replay_buffer.count == args.batch_size:
                if args.alt:
                    alt_agent.update(replay_buffer, cur_steps)
                else:
                    ego_agent.update(replay_buffer, cur_steps)
                replay_buffer.count = 0
            # env.render(interval=0.08)
        if args.use_wandb:
            wandb.log({'episode': k, 'ep_reward': episode_reward})
        print(f'Ep {k} reward:', episode_reward)
    if args.alt:
        alt_agent.save_actor(f'../../models/bcp/bcp_{args.layout}_alt-seed{seed}.pth')
    else:
        ego_agent.save_actor(f'../../models/bcp/bcp_{args.layout}-seed{seed}.pth')


def run():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--use_minibatch", type=bool, default=False, help="whether sample Minibatchs during policy updating")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    # parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--num_episodes',  type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--alt', action='store_true', default=True)

    args = parser.parse_args()
    
    args.max_episode_steps = 600 # Maximum number of steps per episode
    args.t_max = args.num_episodes * args.max_episode_steps
    args.state_dim = 96
    args.action_dim = 6
    name = f'bcp_ppo_{args.layout}_seed{args.seed}'
    if args.alt:
        name = f'bcp_ppo_{args.layout}_seed{args.seed}_alt'
    wandb.init(project='overcooked_rl',
               group='BCP',
               name=name,
               job_type='training',
               config=vars(args),
               dir=os.path.join(WANDB_DIR, 'bcp'),
               reinit=True)


    seed_everything(seed=args.seed)
    ego_agent = PPO_discrete()
    alt_agent = torch.load(BC_MODELS[args.layout], map_location='cpu')
    if args.alt:
        train(args, ego_agent=alt_agent, alt_agent=ego_agent, n_episodes=args.num_episodes, seed=args.seed,
              logger=None)  # wanghm
    else:
        train(args, ego_agent=ego_agent, alt_agent=alt_agent, n_episodes=args.num_episodes, seed=args.seed, logger=None)  # wanghm
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    run()


