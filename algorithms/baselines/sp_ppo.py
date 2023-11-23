import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from My_utils import ReplayBuffer
from agents.ppo_discrete import PPO_discrete
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../PPO-discrete/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import argparse
# warnings.filterwarnings('ignore', message='Passing (type, 1) or \'1type\' as a synonym of type is deprecated')
# from rl_plotter.logger import Logger
from My_utils import seed_everything, LinearAnnealer, init_env
import wandb
from datetime import datetime


def train(args, ego_agent:PPO_discrete, alt_agent:PPO_discrete, n_episodes:int, seed:int):
    annealer = LinearAnnealer(horizon=args.num_episodes * args.max_episode_steps * 0.5)
    env = init_env(layout=args.layout)
    # logger = Logger(exp_name=f'policy_pool/sp_ppo_h32', env_name=args.layout)
    ego_buffer = ReplayBuffer(args.batch_size, args.state_dim)
    alt_buffer = ReplayBuffer(args.batch_size, args.state_dim)
    cur_steps = 0  # Record the total steps during the training
    for k in range(1, n_episodes+1):
        agent_env_steps = args.max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)
        obs  = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
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
            if done:
                dw = True
            else:
                dw = False
            ego_buffer.store(ego_obs, ego_a, ego_a_logprob, r, ego_obs_, dw, done)
            alt_buffer.store(alt_obs, alt_a, alt_a_logprob, r, alt_obs_, dw, done)
            ego_obs = ego_obs_
            alt_obs = alt_obs_
            if ego_buffer.count == args.batch_size:
                ego_agent.update(ego_buffer, cur_steps)
                alt_agent.update(alt_buffer, cur_steps)
                ego_buffer.count = 0
                alt_buffer.count = 0
        print(f"Ep {k}:", episode_reward)
        # wandb.log({'episode': k, 'ep_reward': episode_reward})
        # logger.update(score=[episode_reward], total_steps=k)

        # # save checkpoints of different skill levels
        # if k < 50:
        #     if k % 1 == 0:
        #         ego_agent.save_actor(str(wandb.run.dir) + f"/sp_periodic_{k}.pt")
        # elif k < 100:
        #     if k % 2 == 0:
        #         ego_agent.save_actor(str(wandb.run.dir) + f"/sp_periodic_{k}.pt")
        # else:
        #     if (k % 25 == 0 or k == k - 1):
        #         ego_agent.save_actor(str(wandb.run.dir) + f"/sp_periodic_{k}.pt")


def run():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--use_minibatch", type=bool, default=False, help="whether sample Minibatchs during policy updating")
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='counter_circuit')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    # parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--num_episodes', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    args.max_episode_steps = 600 # Maximum number of steps per episode
    test_env = init_env(layout=args.layout)
    args.state_dim = test_env.observation_space.shape[0]
    args.action_dim = test_env.action_space.n
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    # wandb.init(project='overcooked_rl',
    #            group='FCP',
    #            name=f'sp_ppo_{args.layout}_seed{args.seed}',
    #            job_type='training',
    #            config=vars(args),
    #            reinit=True)

    seed_everything(args.seed)
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
    alt_agent = PPO_discrete(lr=args.lr,
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
    train(args, ego_agent=ego_agent, alt_agent=alt_agent, n_episodes=args.num_episodes, seed=args.seed)

    # wandb.finish()

if __name__ == '__main__':

    run()


