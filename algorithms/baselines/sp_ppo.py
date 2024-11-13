import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from My_utils import ReplayBuffer
from agents.ppo_discrete import PPO_discrete
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../PPO-discrete/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import argparse
# warnings.filterwarnings('ignore', message='Passing (type, 1) or \'1type\' as a synonym of type is deprecated')
from My_utils import seed_everything, LinearAnnealer, init_env, Normalization, RewardScaling
import wandb


def train(args, ego_agent:PPO_discrete, alt_agent:PPO_discrete, n_episodes:int, seed:int):
    annealer = LinearAnnealer(horizon=args.num_episodes * args.max_episode_steps * 0.5)
    env = init_env(layout=args.layout)
    # logger = Logger(exp_name=f'policy_pool/sp_ppo_h32', env_name=args.layout)
    ego_buffer = ReplayBuffer(args.batch_size, args.state_dim)
    alt_buffer = ReplayBuffer(args.batch_size, args.state_dim)
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
            alt_a, alt_a_logprob = alt_agent.choose_action(alt_obs)  # Action and the corresponding log probability
            obs_, sparse_reward, done, info = env.step((ego_a, alt_a))
            shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
            r = sparse_reward + shaped_r * reward_shaping_factor
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
            ego_buffer.store(ego_obs, ego_a, ego_a_logprob, r, ego_obs_, dw, done)
            alt_buffer.store(alt_obs, alt_a, alt_a_logprob, r, alt_obs_, dw, done)
            ego_obs = ego_obs_
            alt_obs = alt_obs_
            if ego_buffer.count == args.batch_size:
                ego_train_info = ego_agent.update(ego_buffer, cur_steps)
                alt_train_info = alt_agent.update(alt_buffer, cur_steps)
                ego_buffer.count = 0
                alt_buffer.count = 0
                if args.use_wandb:
                    log_train(ego_train_info, cur_steps)
            # env.render(interval=0.08)
        print(f"Ep {k}:", episode_reward)
        if args.use_wandb:
            wandb.log({'episode': k, 'ep_reward': episode_reward})
            # save checkpoints of different skill levels
            if k < 50:
                if k % 1 == 0:
                    ego_agent.save_actor(str(wandb.run.dir) + f"/sp_ego_periodic_{k}.pt")
                    alt_agent.save_actor(str(wandb.run.dir) + f"/sp_alt_periodic_{k}.pt")

            elif k < 100:
                if k % 2 == 0:
                    ego_agent.save_actor(str(wandb.run.dir) + f"/sp_ego_periodic_{k}.pt")
                    alt_agent.save_actor(str(wandb.run.dir) + f"/sp_alt_periodic_{k}.pt")
            else:
                """
                使用最后一个episode的模型作为SP agent
                """
                if (k % 25 == 0 or k == k - 1):
                    ego_agent.save_actor(str(wandb.run.dir) + f"/sp_ego_periodic_{k}.pt")
                    alt_agent.save_actor(str(wandb.run.dir) + f"/sp_alt_periodic_{k}.pt")

def log_train(train_infos, total_num_steps):
    for k, v in train_infos.items():
        wandb.log({k: v}, step=total_num_steps)

def run():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--use_minibatch", type=bool, default=False, help="whether sample Minibatchs during policy updating")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    # parser.add_argument("--entropy_coef", type=float, default=0.01)+
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='soup_coordination')
    parser.add_argument('--layout', default='random3')
    # parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--num_episodes', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--use_wandb", action='store_false', default=False)
    args = parser.parse_args()

    args.max_episode_steps = 600 # Maximum number of steps per episode
    test_env = init_env(layout=args.layout)
    args.state_dim = test_env.observation_space.shape[0]
    args.action_dim = test_env.action_space.n
    if args.use_wandb:
        wandb.init(project='overcooked_rl',
                   group='SP_tune22',
                   name=f'sp_ppo_{args.layout}_seed{args.seed}',
                   job_type='training',
                   config=vars(args),
                   reinit=True)

    seed_everything(args.seed)
    ego_agent = PPO_discrete()
    alt_agent = PPO_discrete()
    train(args, ego_agent=ego_agent, alt_agent=alt_agent, n_episodes=args.num_episodes, seed=args.seed)
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    run()


