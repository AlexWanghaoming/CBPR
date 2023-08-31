from agents.agent_utils import ReplayBuffer, Normalization, RewardScaling
from agents.ppo_discrete import PPO_discrete
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../PPO-discrete/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import argparse
# warnings.filterwarnings('ignore', message='Passing (type, 1) or \'1type\' as a synonym of type is deprecated')
# from rl_plotter.logger import Logger
from utils import seed_everything, LinearAnnealer, init_env


def train(args, ego_agent:PPO_discrete, alt_agent:PPO_discrete, n_episodes:int, seed:int, logger):
    annealer = LinearAnnealer(horizon=args.t_max/2)
    env = init_env(layout=args.layout)
    ego_agent.max_train_steps = args.max_episode_steps * n_episodes
    alt_agent.max_train_steps = args.max_episode_steps * n_episodes
    
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    
    ego_buffer = ReplayBuffer(args)
    alt_buffer = ReplayBuffer(args)
    
    cur_steps = 0  # Record the total steps during the training
    
    ego_state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    alt_state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    for k in range(1, n_episodes+1):
        agent_env_steps = args.max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)
        
        obs  = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        if args.use_state_norm:
            ego_obs = ego_state_norm(ego_obs)
            alt_obs = alt_state_norm(alt_obs)
        
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
                alt_obs_ = ego_state_norm(alt_obs_)
            
            elif args.use_reward_scaling:
                r = reward_scaling(r)
            
            if done and episode_steps != args.max_episode_steps:
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
            
            # env.render()
        print(f"Ep {k}:", episode_reward)
        # logger.update(score=[episode_reward], total_steps=total_steps)
    # ego_agent.save_actor(f'{DIR}/models/sp/sp_{args.layout}-seed{seed}.pth')


def run():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--net_arch", type=str, default='mlp', help="policy net arch")
    parser.add_argument("--hidden_width", type=int, default=32, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.98, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=8, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.1, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6: 学习率线性衰减")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip: 0.1")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=bool, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="value function coeffcient")
    
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--layout', default='simple', help='layout name')
    parser.add_argument('--num_episodes',  type=int, default=1500, help='total episodes')
    args = parser.parse_args()

    args.max_episode_steps = 600 # Maximum number of steps per episode
    args.t_max = args.num_episodes * args.max_episode_steps
    
    test_env = init_env(layout=args.layout)
    if args.net_arch == "conv":
        args.state_dim = test_env.observation_space.shape[-1]
    else:
        args.state_dim = test_env.observation_space.shape[0]
    args.action_dim = test_env.action_space.n
    
    seeds = [0, 1, 42, 2022, 2023]
    # seeds = [42]
    for seed in seeds:
        seed_everything(seed=seed)
        ego_agent = PPO_discrete(args)
        alt_agent = PPO_discrete(args)
        # logger = Logger(exp_name=f'SP', env_name=LAYOUT_NAME)
        train(args, ego_agent=ego_agent, alt_agent=alt_agent, n_episodes=args.num_episodes, seed=seed, logger=None)  # wanghm


if __name__ == '__main__':
    run()


