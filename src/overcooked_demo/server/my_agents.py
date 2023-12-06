import sys, os
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../algorithms')
# from experiments.exp1.bpr_RNN_scriptedPolicy import BPR_online, MTPLibrary, NNLibrary, BPR_offline
from experiments.exp1.okr_scriptedPolicy import BPR_online, MTLibrary, AiPolicyLibrary, BPR_offline, MetaTaskLibrary
import argparse
from collections import deque
import numpy as np
import torch
from models import BCP_MODELS, SP_MODELS, FCP_MODELS, META_TASKS


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
        self.args = parser.parse_args()
        self.args.layout = self.og.layouts[0]
        self.Q_len = 5
        self.rho = 0.1
        self.device = 'cuda'

        HPL = MetaTaskLibrary()
        APL = AiPolicyLibrary()
        MTL = MTLibrary()
        bpr_offline = BPR_offline(self.args, HPL=HPL, APL=APL)
        performance_model = bpr_offline.gen_performance_model()
        print("初始performance_model: ", performance_model)
        belief = bpr_offline.gen_belief()
        self.ep_reward = 0
        super(CBPR_ai, self).__init__(agents=APL.gen_policy_library(self.args),
                                      human_policys=HPL.gen_policy_library(tasks=META_TASKS[self.args.layout]),
                                      MT_models=MTL.gen_policy_library(self.args),
                                      performance_model=performance_model,
                                      belief=belief)

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
        self.episode_steps += 1
        # print('trajs:', self.og.trajectory[-1])
        ai_obs, h_obs = self.env.featurize_state_mdp(state)  # OvercookedState -> featurized state
        if self.episode_steps < 5:  # 每一句开始时还没有zeta
            best_agent_id, best_agent = self._reuse_optimal_policy(belief=self.belief)  # 选择belief最大的智能体
        else:
            best_agent_id, best_agent = self._reuse_optimal_policy(belief=self.zeta)  # 选择belief最大的智能体
        ai_act = best_agent.evaluate(ai_obs)  # 智能体选动作
        if len(self.og.joint_action) == 1:  # bug 每一句最开始 joint_action长度是1
            action = Action.ALL_ACTIONS[ai_act]
            return action, None
        _, h_dire = self.og.joint_action
        self.ep_reward += self.og.curr_reward
        h_act = Action.INDEX_TO_ACTION.index(h_dire)
        h_obs = torch.tensor(h_obs, dtype=torch.float32).to(self.device)
        h_act = torch.tensor(np.array([h_act]), dtype=torch.int64).to(self.device)
        self.Q.append((h_obs, h_act))
        self.xi = self._update_xi(self.Q)
        self.zeta = self._update_zeta(t=self.episode_steps, rho=self.rho)
        self.xi =deepcopy(self.zeta)
        action = Action.ALL_ACTIONS[ai_act]
        return action, None

    def reset(self):
        print('RESET GAME！')
        self.episode_steps = 0
        self.Q = deque(maxlen=self.Q_len)
        self.xi = deepcopy(self.belief)

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



# if __name__ == '__main__':
#     ai = CBPR_ai()
#     print('sssss')
