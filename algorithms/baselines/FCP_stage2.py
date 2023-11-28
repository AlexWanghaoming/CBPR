import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from My_utils import ReplayBuffer
from agents.ppo_discrete import PPO_discrete
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../PPO-discrete/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import argparse
import random
# warnings.filterwarnings('ignore', message='Passing (type, 1) or \'1type\' as a synonym of type is deprecated')
# from rl_plotter.logger import Logger
from My_utils import seed_everything, LinearAnnealer, init_env
import wandb
from datetime import datetime

SOURCE_DIR = '/alpha/overcooked_rl'
WANDB_PATH = SOURCE_DIR + '/algorithms/baselines/wandb'
POLICY_POOL_PATH = SOURCE_DIR + "/models/policy_pool"


def train_one_episode(args, env, reward_shaping_factor, ego_agent:PPO_discrete, alt_agent:PPO_discrete, cur_steps, ego_buffer):
    obs = env.reset()
    ego_obs, alt_obs = obs['both_agent_obs']
    episode_steps = 0
    done = False
    episode_reward = 0
    while not done:
        cur_steps += 1
        episode_steps += 1
        ego_a, ego_a_logprob = ego_agent.choose_action(ego_obs)
        alt_a, alt_a_logprob = alt_agent.choose_action(alt_obs)
        obs_, sparse_reward, done, info = env.step((ego_a, alt_a))
        shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
        r = sparse_reward + shaped_r*reward_shaping_factor
        ego_obs_, alt_obs_ = obs_['both_agent_obs']
        episode_reward += r
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
    # logger.update(score=[episode_reward], total_steps=k)
    return episode_reward


def run():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--use_minibatch", type=bool, default=False, help="whether sample Minibatchs during policy updating")
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='cramped_room')
    # parser.add_argument('--layout', default='marshmallow_experiment')
    # parser.add_argument('--layout', default='asymmetric_advantages')
    parser.add_argument('--num_episodes', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()
    args.max_episode_steps = 600 # Maximum number of steps per episode
    test_env = init_env(layout=args.layout)
    args.state_dim = test_env.observation_space.shape[0]
    args.action_dim = test_env.action_space.n
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M") # 年月日小时分钟
    wandb.init(project='overcooked_rl',
               group='FCP',
               tags="fcp",
               # id="aa",
               name=f'fcp_{args.layout}_seed{args.seed}',
               job_type='training',
               config=vars(args),
               reinit=True)
    seed_everything(args.seed)
    ego_agent = PPO_discrete(lr=args.lr,
                         hidden_dim=args.hidden_dim,
                         batch_size=600,
                         use_minibatch=args.use_minibatch,
                         mini_batch_size=args.mini_batch_size,
                         epsilon=args.epsilon,
                         entropy_coef=args.entropy_coef,
                         state_dim=args.state_dim,
                         action_dim=args.action_dim,
                         num_episodes=args.num_episodes,
                         device=args.device)
    filename_list = os.listdir(POLICY_POOL_PATH + f'/{args.layout}/fcp/s1')
    policy_pool_list = [f'{POLICY_POOL_PATH}/{args.layout}/fcp/s1/{pt}' for pt in filename_list]
    annealer = LinearAnnealer(horizon=args.num_episodes * args.max_episode_steps * 0.5)
    env = init_env(layout=args.layout)
    # logger = Logger(exp_name=f'policy_pool/sp_ppo_h32', env_name=args.layout)
    ego_buffer = ReplayBuffer(args.batch_size, args.state_dim)
    cur_steps = 0  # Record the total steps during the training
    for k in range(1, args.num_episodes + 1):
        partner_policy_path = random.choice(policy_pool_list)
        print('switch partner:', partner_policy_path.split('/')[-1])
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
        alt_agent.load_actor(model_path=partner_policy_path)
        agent_env_steps = args.max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)

        episode_reward = train_one_episode(args,
                                          env=env,
                                          reward_shaping_factor=reward_shaping_factor,
                                          ego_agent=ego_agent,
                                          alt_agent=alt_agent,
                                          cur_steps=cur_steps,
                                          ego_buffer=ego_buffer)
        cur_steps += args.max_episode_steps
        print(f"Ep {k}:", episode_reward)
        wandb.log({'episode': k, 'ep_reward': episode_reward})
    ego_agent.save_actor(f"../../models/fcp/fcp_{args.layout}_seed{args.seed}.pt")
    wandb.finish()


if __name__ == '__main__':
    run()


