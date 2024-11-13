import numpy as np
import torch
from agents.ppo_discrete import PPO_discrete
from models import BC_MODELS, BCP_MODELS
import torch.nn as nn
from bc.bc_hh import BehaviorClone
from My_utils import *
from typing import *
import pickle


def collect_trajs(args, ego_agent:PPO_discrete, alt_agent:nn.Module) -> Dict[Tuple,Dict[str, np.ndarray]]:
    N = 50000   # 每个 meta-task需要收集 N 条数据用于训练 GP
    meta_task_trajs = {}
    meta_task_steps = {}
    env = init_env(layout=args.layout, lossless_state_encoding=False)
    for k in range(1, args.num_episodes + 1):
        obs = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        done = False
        episode_reward = 0
        while not done:
            key = tuple(alt_obs[slice(4, 8)])
            if key not in meta_task_trajs:
                meta_task_trajs[key] = {'train_s':[], 'train_a':[], 'train_r':[], 'train_s_':[]}
                meta_task_steps[key] = 0
            meta_task_steps[key] += 1

            ego_a, ego_a_logprob = ego_agent.choose_action(ego_obs)
            alt_a = alt_agent.choose_action(alt_obs, deterministic=True)

            obs_, sparse_reward, done, info = env.step((ego_a, alt_a))
            ego_obs_, alt_obs_ = obs_['both_agent_obs']
            episode_reward += sparse_reward

            if meta_task_steps[key] <= N:
                meta_task_trajs[key]['train_s'].append(alt_obs)
                meta_task_trajs[key]['train_a'].append(alt_a)
                meta_task_trajs[key]['train_r'].append(sparse_reward)
                meta_task_trajs[key]['train_s_'].append(alt_obs_)

            ego_obs = ego_obs_
            alt_obs = alt_obs_

        # print(f"Ep {k} test reward:", episode_reward)

        # 当所有meta-task的数据收集齐之后，退出循环
        if k > 15000:
            if all(value >= N for value in meta_task_steps.values()):
                break
    for key in meta_task_trajs:
        for i in meta_task_trajs[key]:
            meta_task_trajs[key][i] = np.array(meta_task_trajs[key][i])

    return meta_task_trajs


if __name__ == "__main__":

    default_args = parse_args()
    args = default_args
    # args.layout = "cramped_room"
    args.layout = 'marshmallow_experiment'

    args.num_episodes = 20000
    test_env = init_env(layout=args.layout, lossless_state_encoding=False)
    args.state_dim = test_env.observation_space.shape[0]
    args.action_dim = test_env.action_space.n

    ai = PPO_discrete(args)
    ai.load_actor(BCP_MODELS[args.layout])
    h = torch.load(BC_MODELS[args.layout], map_location='cpu')

    meta_task_trajs = collect_trajs(args, ego_agent=ai, alt_agent=h)
    f_save = open(f'gp_trajs_{args.layout}.pkl', 'wb')
    pickle.dump(meta_task_trajs, f_save)
    f_save.close()