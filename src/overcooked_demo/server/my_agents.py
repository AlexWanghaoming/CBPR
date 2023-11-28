import sys, os
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../algorithms')
from experiments.exp1.bpr_NN_scriptedPolicy import BPR_online, MTPLibrary, NNLibrary, BPR_offline
import argparse
from collections import deque
import numpy as np
import torch
from models import BCP_MODELS, SP_MODELS, FCP_MODELS


class CBPR_ai(BPR_online):
    def __init__(self, og):
        self.og  = og
        agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": self.og.layouts[0]},
            env_params={
                "horizon": 600
            },  # Defining the horizon of the mdp of origin of the trajectories
        )
        self.mdp = agent_evaluator.env.mdp
        self.env = agent_evaluator.env
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description='''Bayesian policy reuse algorithm on overcooked''')
        parser.add_argument('--Q_len', type=int, default=25)
        self.args = parser.parse_args()
        self.args.layout = self.og.layouts[0]
        self.device = 'cuda'

        APL = MTPLibrary()
        mtp_lib = APL.gen_policy_library(args=self.args)  # 构建AI策略库
        NNL = NNLibrary()
        NN_models = NNL.gen_policy_library(self.args)
        bpr_offline = BPR_offline(self.args)
        belief = bpr_offline.gen_belief()
        self.ep_reward = 0
        super(CBPR_ai, self).__init__(agents=mtp_lib, human_policys=None, NN_models=NN_models, belief=belief)

    # debug 用于测试
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
        ai_obs, h_obs = self.env.featurize_state_mdp(state)  # OvercookedState -> featurized state
        best_agent_id, best_agent = self._reuse_optimal_policy()  # 选择belief最大的智能体
        ai_act = best_agent.evaluate(ai_obs)  # 智能体选动作

        if len(self.og.joint_action) == 1:  #
            action = Action.ALL_ACTIONS[ai_act]
            return action, None

        _, h_dire = self.og.joint_action
        h_act = Action.ALL_ACTIONS.index(h_dire)
        next_state, info = self.mdp.get_state_transition(state, self.og.joint_action)  # 手动进行一次状态转移
        self.ep_reward += self.og.curr_reward
        # print('cummulated ep_reward:', self.ep_reward)
        _, h_obs_ = self.env.featurize_state_mdp(next_state)
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
        print('RESET GAME！')
        self.Q = deque(maxlen=self.args.Q_len)


class Baseline_ai():
    def __init__(self, og, agent_type:str):
        self.og  = og
        self.layout = self.og.layouts[0]
        print('Current layout:', self.og.layouts[0])
        agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": self.layout},
            env_params={
                "horizon": 600
            },
        )
        self.mdp = agent_evaluator.env.mdp
        self.env = agent_evaluator.env
        self.device = 'cuda'
        self.ep_reward = 0
        if agent_type == 'BCP':
            self.ai_agent = torch.load(BCP_MODELS[self.layout])
        elif agent_type == 'FCP':
            self.ai_agent = torch.load(FCP_MODELS[self.layout])
        elif agent_type == "SP":
            self.ai_agent = torch.load(SP_MODELS[self.layout])
        else:
            pass

    def evaluate(self, actor, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = actor(s).detach().cpu().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def action(self, state):
        # print('trajs:', self.og.trajectory[-1])
        ai_obs, h_obs = self.env.featurize_state_mdp(state)  # OvercookedState -> featurized state
        ai_act = self.evaluate(self.ai_agent, ai_obs)
        action = Action.ALL_ACTIONS[ai_act]
        return action, None

    def reset(self):
        print('RESET GAME！')


class FCP_ai():
    def __init__(self, og):
        self.og  = og
        self.layout = self.og.layouts[0]
        print('Current layout:', self.og.layouts[0])
        agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": self.layout},
            env_params={
                "horizon": 600
            },
        )
        self.mdp = agent_evaluator.env.mdp
        self.env = agent_evaluator.env
        self.device = 'cuda'
        self.ep_reward = 0
        self.ai_agent = torch.load(FCP_MODELS[self.layout])

    def evaluate(self, actor, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = actor(s).detach().cpu().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def action(self, state):
        # print('trajs:', self.og.trajectory[-1])
        ai_obs, h_obs = self.env.featurize_state_mdp(state)  # OvercookedState -> featurized state
        ai_act = self.evaluate(self.ai_agent, ai_obs)
        action = Action.ALL_ACTIONS[ai_act]
        return action, None

    def reset(self):
        print('RESET GAME！')


class SP_ai():
    def __init__(self, og):
        self.og  = og
        self.layout = self.og.layouts[0]
        print('Current layout:', self.og.layouts[0])
        agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": self.layout},
            env_params={
                "horizon": 600
            },
        )
        self.mdp = agent_evaluator.env.mdp
        self.env = agent_evaluator.env
        self.device = 'cuda'
        self.ep_reward = 0
        self.ai_agent = torch.load(BCP_MODELS[self.layout])

    def evaluate(self, actor, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = actor(s).detach().cpu().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def action(self, state):
        # print('trajs:', self.og.trajectory[-1])
        ai_obs, h_obs = self.env.featurize_state_mdp(state)  # OvercookedState -> featurized state
        ai_act = self.evaluate(self.ai_agent, ai_obs)
        action = Action.ALL_ACTIONS[ai_act]
        return action, None

    def reset(self):
        print('RESET GAME！')


# if __name__ == '__main__':
#     ai = CBPR_ai()
#     print('sssss')
