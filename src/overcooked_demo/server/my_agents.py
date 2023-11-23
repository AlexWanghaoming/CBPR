import sys, os
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction

import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../algorithms')
from bpr_NN_scriptedPolicy import BPR_online, MTPLibrary, NNLibrary, BPR_offline
import argparse
from collections import deque
import numpy as np
import torch


class CBPR_ai(BPR_online):
    def __init__(self, og):
        agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": 'cramped_room'},
            env_params={
                "horizon": 600
            },  # Defining the horizon of the mdp of origin of the trajectories
        )
        self.og  = og
        self.mdp = agent_evaluator.env.mdp
        self.env = agent_evaluator.env

        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description='''Bayesian policy reuse algorithm on overcooked''')
        parser.add_argument('--layout', default='cramped_room')
        parser.add_argument('--Q_len', type=int, default=25)
        parser.add_argument('--plot_group', type=str, default='BPR_NN_MOD')
        self.args = parser.parse_args()
        self.device = 'cuda'
        APL = MTPLibrary()
        mtp_lib = APL.gen_policy_library(args=self.args)  # 构建AI策略库
        NNL = NNLibrary()
        NN_models = NNL.gen_policy_library(self.args)
        bpr_offline = BPR_offline(self.args)
        belief = bpr_offline.gen_belief()
        self.ep_reward = 0
        super(CBPR_ai, self).__init__(agents=mtp_lib,
                                human_policys=None,
                                NN_models=NN_models,
                                belief=belief)

    # debug
    # def action(self, state):
    #     obs = self.env.featurize_state_mdp(state)
    #     [action] = random.sample(
    #         [
    #             # Action.STAY,
    #             # Direction.NORTH,
    #             # Direction.SOUTH,
    #             Direction.WEST,
    #             Direction.EAST,
    #             # Action.INTERACT,
    #         ],
    #         1,
    #     )
    #     return action, None

    def action(self, state):
        # print('trajs:', self.og.trajectory[-1])
        print('curr_rew:', self.og.curr_reward)
        ai_obs, h_obs = self.env.featurize_state_mdp(state)
        best_agent_id, best_agent = self._reuse_optimal_policy()  # 选择belief最大的智能体
        ai_act = best_agent.evaluate(ai_obs)  # 智能体选动作
        if len(self.og.joint_action) == 1:  #
            action = Action.ALL_ACTIONS[ai_act]
            return action, None
        _, h_dire = self.og.joint_action
        h_act = Action.ALL_ACTIONS.index(h_dire)
        print(Action.ALL_ACTIONS)
        next_state, info = self.mdp.get_state_transition(
            state, self.og.joint_action
        )
        self.ep_reward += self.og.curr_reward
        print('cummulated ep_reward:', self.ep_reward)
        ai_obs_, h_obs_ = self.env.featurize_state_mdp(next_state)
        actions_one_hot = np.eye(6)[h_act]
        obs_x = np.hstack([h_obs, actions_one_hot])
        obs_x = torch.from_numpy(obs_x).float().to(self.device)
        obs_y = np.hstack([h_obs_, self.og.curr_reward])  # s_prime, r
        obs_y = torch.from_numpy(obs_y).float().to(self.device)
        self.Q.append((obs_x, obs_y))
        self.belief = self._update_beta(self.Q)
        action = Action.ALL_ACTIONS[ai_act]
        return action, None

    def reset(self):
        self.Q = deque(maxlen=self.args.Q_len)
        print('ep_reward:', self.ep_reward)










# if __name__ == '__main__':
#     ai = CBPR_ai()
#     print('sssss')
