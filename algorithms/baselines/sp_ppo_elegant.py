import sys
from argparse import ArgumentParser
import os
import sys
import time
import torch
import numpy as np
import torch.multiprocessing as mp  # torch.multiprocessing extends multiprocessing of Python
from copy import deepcopy
from multiprocessing import Process, Pipe

from elegantrl.train.config import build_env
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.evaluator import Evaluator, get_cumulative_rewards_and_steps
from typing import *
from torch import Tensor
sys.path.append("../..")
if True:  # write after `sys.path.append("..")`
    # from elegantrl import train_agent, train_agent_multiprocessing
    from elegantrl import Config, get_gym_env_args
    from elegantrl.agents import AgentDiscretePPO
import gym
from utils import init_env

def explore_one_env(ego_agent, alt_agent, env, horizon_len: int):

    ego_states = torch.zeros((horizon_len, ego_agent.num_envs, ego_agent.state_dim), dtype=torch.float32).to(ego_agent.device)
    alt_states = torch.zeros((horizon_len, alt_agent.num_envs, alt_agent.state_dim), dtype=torch.float32).to(alt_agent.device)
    ego_actions = torch.zeros((horizon_len, ego_agent.num_envs, 1), dtype=torch.float32).to(ego_agent.device)
    alt_actions = torch.zeros((horizon_len, alt_agent.num_envs, 1), dtype=torch.float32).to(alt_agent.device)
    ego_logprobs = torch.zeros(horizon_len, ego_agent.num_envs, dtype=torch.float32).to(ego_agent.device)
    alt_logprobs = torch.zeros(horizon_len, alt_agent.num_envs, dtype=torch.float32).to(alt_agent.device)
    rewards = torch.zeros(horizon_len, ego_agent.num_envs, dtype=torch.float32).to(ego_agent.device)
    dones = torch.zeros(horizon_len, ego_agent.num_envs, dtype=torch.bool).to(ego_agent.device)

    ego_state = ego_agent.last_state  # state.shape == (1, state_dim) for a single env.
    alt_state = alt_agent.last_state

    get_action1 = ego_agent.act.get_action
    get_action2 = alt_agent.act.get_action

    convert = ego_agent.act.convert_action_for_env

    for t in range(horizon_len):
        ego_action, ego_logprob = [t.squeeze(0) for t in get_action1(ego_state.unsqueeze(0))[:2]]
        alt_action, alt_logprob = [t.squeeze(0) for t in get_action2(alt_state.unsqueeze(0))[:2]]
        ego_states[t] = ego_state
        alt_states[t] = alt_state

        ego_ary_action = convert(ego_action).item()
        alt_ary_action = convert(alt_action).item()

        ary_state, sparse_reward, done, info = env.step((ego_ary_action, alt_ary_action))  # next_state
        reward_shaping_factor = 0
        shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
        r = sparse_reward + shaped_r * reward_shaping_factor

        ary_state = env.reset() if done else ary_state
        ego_ary_state, alt_ary_state = ary_state['both_agent_obs']

        ego_state = torch.as_tensor(ego_ary_state,
                                dtype=torch.float32,
                                device=ego_agent.device).unsqueeze(0)
        alt_state = torch.as_tensor(alt_ary_state,
                                dtype=torch.float32,
                                device=alt_agent.device).unsqueeze(0)

        ego_actions[t] = ego_action
        alt_actions[t] = alt_action
        ego_logprobs[t] = ego_logprob
        alt_logprobs[t] = alt_logprob
        rewards[t] = r
        dones[t] = done

    ego_agent.last_state = ego_state  # state.shape == (1, state_dim) for a single env.
    alt_agent.last_state = alt_state  # state.shape == (1, state_dim) for a single env.

    rewards *= ego_agent.reward_scale
    undones = 1.0 - dones.type(torch.float32)
    return [(ego_states, ego_actions, ego_logprobs, rewards, undones),
            (alt_states, alt_actions, alt_logprobs, rewards, undones)]


def train_agent(args: Config):
    args.init_before_training()
    torch.set_grad_enabled(False)

    '''init environment'''
    env = init_env(layout='cramped_room', lossless_state_encoding=False)
    '''init agent'''
    ego_agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    ego_agent.save_or_load_agent(args.cwd, if_save=False)

    alt_agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    alt_agent.save_or_load_agent(args.cwd, if_save=False)

    '''init agent.last_state'''
    state = env.reset()
    ego_state, alt_state = state['both_agent_obs']

    ego_state = torch.tensor(ego_state, dtype=torch.float32, device=ego_agent.device).unsqueeze(0)
    alt_state = torch.tensor(alt_state, dtype=torch.float32, device=alt_agent.device).unsqueeze(0)
    # if args.num_envs == 1:
    #     assert state[0].shape == (args.state_dim,)
    #     assert isinstance(state[0], np.ndarray)
    #     state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    # else:
    #     assert state.shape == (args.num_envs, args.state_dim)
    #     assert isinstance(state, torch.Tensor)
    #     state = state.to(agent.device)
    # assert state.shape == (args.num_envs, args.state_dim)
    # assert isinstance(state, torch.Tensor)
    ego_agent.last_state = ego_state.detach()
    alt_agent.last_state = alt_state.detach()


    buffer = []

    '''init evaluator'''
    eval_env =  init_env(layout='cramped_room', lossless_state_encoding=False)
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=False)

    '''train loop'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_save_buffer = args.if_save_buffer

    if_train = True
    total_step = 0
    while if_train:
        for agent, buffer_items in zip([ego_agent, alt_agent],
                                       explore_one_env(ego_agent, alt_agent, env, horizon_len)):
            total_step += horizon_len
            exp_r = buffer_items[2].mean().item()

            buffer[:] = buffer_items

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            ## 模型评估
            evaluator.evaluate_and_save(ego_actor=ego_agent.act,
                                        alt_actor=alt_agent.act,
                                        steps=horizon_len,
                                        exp_r=exp_r,
                                        logging_tuple=logging_tuple)

            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    env.close() if hasattr(env, 'close') else None
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
        buffer.save_or_load_history(cwd, if_save=True)


if __name__ == '__main__':
    agent_class = AgentDiscretePPO  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Overcooked-v0',
        # 'max_step': 500,
        'state_dim': 96,
        'action_dim': 6,
        'max_step':  600*1500,
        'if_discrete': True,
    }
    # get_gym_env_args(env=gym.make('CartPole-v1'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = 2048
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -2
    args.learning_rate = 2e-5
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_times = 32
    args.eval_per_step = 1e4

    args.gpu_id = 0
    args.num_workers = 4
    # train_agent_multiprocessing(args)
    train_agent(args)