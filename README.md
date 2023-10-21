
## Introduction 🥘
Popular RL algorithms based [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) environment (6bde8b27b5a1dcdba571e8f53d98c6fc836eca8c).
Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).


## Code Structure Overview 🗺
`bc/`:
- `bc_hh.py`: 用行为克隆拟合当前layout下的人类策略
- `bc_hh_meta_task.py`: 根据key state划分人类轨迹,分别拟合不同的meta-task models

`algorithms/`:
- `bcp.py`: 根据BC模型训练BCP模型
- `bpr_NN.py`和`bpr_gp.py`: 复现论文 *Efficient Bayesian Policy Reuse With a Scalable Observation Model in Deep Reinforcement Learning* 算法
- `mtp.py`: 训练与不同meta-task models合作的BCP模型
- `okr.py`: 复现论文 *Accurate policy detection and efficient knowledge reuse against multi-strategic opponents*算法
- `sp_ppo.py`:  基于PPO的自博弈训练, state normalization 方法来自 elegantRL
- `sp_sac.py`: 基于SAC的自博弈训练，超参数设置 :
`{'hidden_dim': 64, 'lr': 1e-4, 'tau': 0.005, 'adaptive_alpha': True, 'clip_grad_norm': 0.1, 'use_lr_decay': False, 'buffer_size': 1e6, 'batch_size': 256}`

`state_trans_func/`:
- `collect_trajs.py`: 收集每一种meta-task的轨迹
- `GP_GPy.py`: 用GPy拟合 不同meta-tasks的状态转移函数（已弃用）
- `GP_gpytorch.py`: 用gpytorch拟合 不同meta-tasks的状态转移函数
- `NN.py`: 用神经网络拟合不同meta-tasks的状态转移函数

`src/overcooked_ai_py/` contains:
`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically


