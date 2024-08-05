import argparse
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agents/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from models import BCP_MODELS, SP_MODELS, FCP_MODELS, SP_ALT_MODELS,BCP_ALT_MODELS
from My_utils import seed_everything, init_env, evaluate_actor, print_mean_interval
import wandb
from algorithms.cbpr import CBPR

WANDB_DIR = '/alpha/overcooked_rl/my_wandb_log'
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--layout', default='soup_coordination')
    # parser.add_argument('--layout', default='coordination_ring')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=600)
    parser.add_argument('--Q_len', type=int, default=20)
    parser.add_argument('--rho', type=float, default=0.1,
                        help="a hyperparameter which controls the weight of the inter-episode and intra-episode beliefs")
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--use_wandb', action='store_true', default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    seed_everything(args.seed)
    env = init_env(layout=args.layout)
    ego_agents = {'BCP': torch.load(BCP_MODELS[args.layout], map_location='cpu'),
              'FCP':torch.load(FCP_MODELS[args.layout], map_location='cpu'),
              'SP':torch.load(SP_MODELS[args.layout], map_location='cpu'),
              'CBPR':CBPR(args)}
    alt_agents = {'BCP': torch.load(BCP_MODELS[args.layout], map_location='cpu'),
               'FCP': torch.load(FCP_MODELS[args.layout], map_location='cpu'),
               'SP': torch.load(SP_MODELS[args.layout], map_location='cpu'),
               'CBPR': CBPR(args)}
    if args.use_wandb:
        wandb.init(project='overcooked_rl',
                   group='exp5',
                   name=f'{args.layout}',
                   config=vars(args),
                   job_type='eval',
                   dir=os.path.join(WANDB_DIR, 'exp5'),
                   reinit=True)
        wandb_tab = wandb.Table(columns=list(ego_agents.keys()))

    for ego in ego_agents:
        mean = []
        sigma = []
        for alt in alt_agents:
            ai_agent1 = ego_agents[ego]
            ai_agent2 = alt_agents[alt]

            r_list = []
            for k in range(args.num_episodes):
                obs = env.reset()
                ai_obs, h_obs = obs['both_agent_obs']
                ep_reward = 0
                done = False
                episode_steps = 0
                info = {}
                while not done:
                    episode_steps += 1
                    if isinstance(ai_agent1, CBPR):
                        ai_act = ai_agent1.predict(ego_obs=ai_obs,
                                                   alt_obs=h_obs,
                                                   info=info,
                                                   ep_reward=ep_reward,
                                                   deterministic=True)
                    else:
                        ai_act = evaluate_actor(ai_agent1, ai_obs, deterministic=True)

                    if isinstance(ai_agent2, CBPR):
                        h_act = ai_agent2.predict(ego_obs=h_obs,
                                                  alt_obs=ai_obs,
                                                  info=info,
                                                  ep_reward=ep_reward,
                                                  deterministic=False)
                    else:
                        h_act = evaluate_actor(ai_agent2, h_obs, deterministic=False)

                    obs, sparse_reward, done, info = env.step((ai_act, h_act))
                    # obs, sparse_reward, done, info = env.step((h_act, ai_act))
                    ai_obs, h_obs = obs['both_agent_obs']
                    ep_reward += sparse_reward
                    # env.render(interval=0.05)
                # print(f'Ep {k + 1}:', ep_reward)
                r_list.append(ep_reward)
            m,s = print_mean_interval(r_list)
            mean.append(m)
            sigma.append(s)
        wandb_tab.add_data(*mean)
                # if args.use_wandb:
                #     wandb.log({'episode': k + 1, 'ep_reward': ep_reward})
    if args.use_wandb:
        wandb.finish()