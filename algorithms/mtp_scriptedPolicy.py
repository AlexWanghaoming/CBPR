import argparse
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from agents.ppo_discrete import PPO_discrete
from My_utils import seed_everything, LinearAnnealer, init_env, ReplayBuffer, RewardScaling
import wandb

# add = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = add
# os.environ['https_proxy'] = add
WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'


def train(args, ego_agent:PPO_discrete, n_episodes:int, seed:int, logger):
    annealer = LinearAnnealer(horizon=args.num_episodes * args.max_episode_steps * 0.5)

    env = init_env(layout=args.layout,
                   agent0_policy_name='mtp',
                   agent1_policy_name=f'script:{args.scripted_policy_name}',
                   use_script_policy=True)

    ego_buffer = ReplayBuffer(args.batch_size, args.state_dim)
    cur_steps = 0  # Record the total steps during the training
    if args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    for k in range(1, n_episodes+1):
        agent_env_steps = args.max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)
        obs  = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        if args.use_reward_scaling:
            reward_scaling.reset()

        episode_steps = 0
        done = False
        episode_reward = 0
        while not done:
            cur_steps += 1
            episode_steps += 1
            ego_a, ego_a_logprob = ego_agent.choose_action(ego_obs)
            obs_, sparse_reward, done, info = env.step((ego_a, 0))
            shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
            r = sparse_reward + shaped_r*reward_shaping_factor
            ego_obs_, alt_obs_ = obs_['both_agent_obs']
            episode_reward += r
            if args.use_reward_scaling:
                r = reward_scaling(r)
            if done:
                dw = True
            else:
                dw = False
            ego_buffer.store(ego_obs, ego_a, ego_a_logprob, r, ego_obs_, dw, done)
            ego_obs = ego_obs_
            alt_obs = alt_obs_
            # env.render(interval=0.08)
            if ego_buffer.count == args.batch_size:
                ego_agent.update(ego_buffer, cur_steps)
                ego_buffer.count = 0
        wandb.log({'episode': k, 'ep_reward': episode_reward})
        # print(f"Ep {k}:", episode_reward)
    ego_agent.save_actor(f'../models/mtp/mtp_{args.layout}-{args.scripted_policy_name}-seed{seed}.pth')


def run():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--use_minibatch", type=bool, default=False, help="whether sample Minibatchs during policy updating")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='coordination_ring')
    parser.add_argument('--layout', default='soup_coordination')
    # parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--num_episodes',  type=int, default=2000)
    parser.add_argument('--scripted_policy_name', type=str, default='random')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args.max_episode_steps = 600 # Maximum number of steps per episode
    args.t_max = args.num_episodes * args.max_episode_steps
    args.state_dim = 96
    args.action_dim = 6
    wandb.init(project='overcooked_rl',
               group='MTP',
               name=f'mtp_ppo_{args.layout}_{args.scripted_policy_name}_seed{args.seed}',
               job_type='training',
               config=vars(args),
               dir=os.path.join(WANDB_DIR, 'mtp'),
               reinit=True)

    seed_everything(seed=args.seed)
    ego_agent = PPO_discrete(num_episodes=args.num_episodes, device=args.device)
    train(args, ego_agent=ego_agent, n_episodes=args.num_episodes, seed=args.seed, logger=None)
    wandb.finish()


if __name__ == '__main__':
    run()


