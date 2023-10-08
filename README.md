
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
- `sp.py`:  基于PPO的自博弈算法

`state_trans_func/`:
- `collect_trajs.py`: 收集每一种meta-task的轨迹
- `GP_GPy.py`: 用GPy拟合 不同meta-tasks的状态转移函数（已弃用）
- `GP_gpytorch.py`: 用gpytorch拟合 不同meta-tasks的状态转移函数
- `NN.py`: 用神经网络拟合不同meta-tasks的状态转移函数

`overcooked_ai_py/` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning/`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

`human_aware_rl/` contains:

`ppo/`:
- `ppo_rllib.py`: Primary module where code for training a PPO agent resides. This includes an rllib compatible wrapper on `OvercookedEnv`, utilities for converting rllib `Policy` classes to Overcooked `Agent`s, as well as utility functions and callbacks
- `ppo_rllib_client.py` Driver code for configuing and launching the training of an agent. More details about usage below
- `ppo_rllib_from_params_client.py`: train one agent with PPO in Overcooked with variable-MDPs 
- `ppo_rllib_test.py` Reproducibility tests for local sanity checks
- `run_experiments.sh` Script for training agents on 5 classical layouts
- `trained_example/` Pretrained model for testing purposes

`rllib/`:
- `rllib.py`: rllib agent and training utils that utilize Overcooked APIs
- `utils.py`: utils for the above
- `tests.py`: preliminary tests for the above

`imitation/`:
- `behavior_cloning_tf2.py`:  Module for training, saving, and loading a BC model
- `behavior_cloning_tf2_test.py`: Contains basic reproducibility tests as well as unit tests for the various components of the bc module.

`human/`:
- `process_data.py` script to process human data in specific formats to be used by DRL algorithms
- `data_processing_utils.py` utils for the above

`utils.py`: utils for the repo

`overcooked_demo` contains:

`server/`:
- `app.py`: The Flask app 
- `game.py`: The main logic of the game. State transitions are handled by overcooked.Gridworld object embedded in the game environment
- `move_agents.py`: A script that simplifies copying checkpoints to [agents](src/overcooked_demo/server/static/assets/agents/) directory. Instruction of how to use can be found inside the file or by running `python move_agents.py -h`

`up.sh`: Shell script to spin up the Docker server that hosts the game 



