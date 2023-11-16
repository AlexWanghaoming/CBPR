## Introduction 🥘
Popular RL algorithms based [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) environment (6bde8b27b5a1dcdba571e8f53d98c6fc836eca8c).
Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).
旧版本的overcooked中，三个洋葱放入锅中后会自动开始烹饪，而新版本overcooked的游戏逻辑中，智能体将原材料放入锅中后，需要再执行一次‘interact’动作后才开始烹饪。
数据CLEAN_2019_HUMAN_DATA_TRAIN和CLEAN_2020_HUMAN_DATA_TRAIN分别根据旧版本和新版本的逻辑采集，为了使数据通用，本仓库已经将游戏设置修改回旧版本。

## Code Structure Overview 🗺
`bc/`:
- `bc_hh.py`: behavior cloning using all collected human trajectories over a specific layout, note: human trajectories were divided into two groups (i.e. BC and HP)
- `bc.sh`: run `bc_hh.py` in for-loop
- `bc_hh_meta_task.py`: 根据key state划分人类轨迹,训练 meta-task models
- `replay_human_data.py`: 回放CLEAN_2019_HUMAN_DATA_TRAIN和CLEAN_2020_HUMAN_DATA_TRAIN中的人类轨迹

`algorithms/` contains: 
- `baselines/`:
    - `script_agent_playing.py`: 测试 script agents 合作性能
    - `sp_ppo.py`:  基于PPO的自博弈训练
    - `train_sp.sh`: FCP的stage1, 训练多个sp合作智能体
    - `sp_sac.py`: 基于SAC的自博弈训练，超参数设置 : `{'hidden_dim': 64, 'lr': 1e-4, 'tau': 0.005, 'adaptive_alpha': True, 'clip_grad_norm': 0.1, 'use_lr_decay': False, 'buffer_size': 1e6, 'batch_size': 256}`
    - `bcp.py`: Behavioral Cloning Play 复现论文 *On the Utility of Learning about Humans for Human-AI Coordination, nips 2019* 算法
    - `FCP_stage1.py`和`FCP_stage2.py`: 两阶段FCP 复现论文*Collaborating with human without human data, nips 2021* 算法
    - `evaluate_bcp*.py`: 评估BCP和切换策略的meta-task model合作表现
- `bpr_NN.py`和`bpr_gp.py`: 复现论文 *Efficient Bayesian Policy Reuse With a Scalable Observation Model in Deep Reinforcement Learning, TNNLS* 算法
- `mtp*.py`: 训练与不同meta-task models合作的BCP模型
- `okr.py`: 复现论文 *Accurate policy detection and efficient knowledge reuse against multi-strategic opponents* 算法

`state_trans_func/`:
- `collect_trajs.py`: 收集每一种meta-task的轨迹
- `GP_GPy.py`: 用GPy拟合 不同meta-tasks的状态转移函数（已弃用）
- `GP_gpytorch.py`: 用gpytorch拟合 不同meta-tasks的状态转移函数
- `NN.py`: 神经网络拟合不同meta-tasks的状态转移函数

`src/overcooked_ai_py/` contains:
`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`script_agent` from *Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased, ICLR 2023*:
- avaliable rule-based policy in overcooked

`miscs` contains:
- `plot_*_distribution.py` 计算并绘制hh（人类+人类），BC-hh(BC+人类)，BCP-BC，BCP-hh(BCP+人类)  游戏轨迹分布并计算分布差异