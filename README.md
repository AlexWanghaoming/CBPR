# 🥘 CBPR 
Official code for NeurIPS 2024 paper: *Beyond Single Stationary Policies: Meta-Task Players as Naturally Superior Collaborators*
# Installation
```
conda create -n cbpr python=3.7
conda activate cbpr
pip install -r requirements.txt
```
# Training
Training of all baseline algorithm and offline stage of CBPR. The trained models are in `models`
## Self-play
Run `./algorithms/baselines/train_sp.sh` to train SP agent in _Cramped Room_ layout.
## FCP
FCP agent was introduced in [Collaborating with Humans without Human Data](https://arxiv.org/abs/2110.08176). To build policy pool, run `python FCP_stage1.py`. To train FCP agent, please run `python FCP_stage2.py`.
## BCP
BCP agent was introduced in [On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789). To train BCP agent, we firstly train behavioral cloning model using `./bc/bc.sh`. Next, train BCP agent using `./algorithms/baselines/train_bcp.sh`.
## MTP
During the offline stage of CBPR, we train MTP agents by pairing them with rule-based agents using `./algorithm/mtp_scriptedPolicy.sh`.

# Evaluation
## Collaborating with agents that switch policies
Pair BCP agent with agent that switches policies every 100 timesteps in _Cramped Room_ layout. 
```
python experiments/exp1/evaluate_scriptPolicy.py --layout cramped_room --num_episodes 50 --mode intra --switch_human_freq 100 --seed 1 --algorithm BCP
```
Pair FCP agent with agent that switches policies every 2 episodes in _Cramped Room_ layout. 
```
python experiments/exp1/evaluate_scriptPolicy.py --layout cramped_room --num_episodes 50 --mode inter --switch_human_freq 2 --seed 1 --algorithm FCP
```
Pair CBPR with agent that switches policies every 200 timesteps in _Cramped Room_ layout. 
```
python experiments/exp1/okr_scriptedPolicy.py --layout cramped_room --num_episodes 50 --mode intra --switch_human_freq 200 --seed 1 --Q_len 20 --rho 0.1
```
## Collaborating with agents using various skill levels
Pair BCP agent with agent using _high_ skill level in _Cramped Room_ layouts.
```
python experiments/exp2/evaluate_skill_levels.py --layout cramped_room --num_episodes 50 --skill_level high --algorithm BCP --use_wandb
```
Pair FCP agent with agent using high _low_ level in _Cramped Room_ layouts.
```
python experiments/exp2/evaluate_skill_levels.py --layout cramped_room --num_episodes 50 --skill_level low --algorithm FCP --use_wandb
```
Pair CBPR with agent using _medium_ skill level in _Cramped Room_ layouts.
```
python experiments/exp2/okr_skill_levels.py --layout cramped_room --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.1 --use_wandb
```
## Run human-AI test, display in 127.0.0.1:5001
``` 
python src/overcooked_demo/server/app.py
```

# Citation
If you find this repository useful, please cite these papers:
```
@article{carroll2019utility,
  title={On the utility of learning about humans for human-ai coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
@article{strouse2021collaborating,
  title={Collaborating with humans without human data},
  author={Strouse, DJ and McKee, Kevin and Botvinick, Matt and Hughes, Edward and Everett, Richard},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={14502--14515},
  year={2021}
}
@article{yu2023learning,
  title={Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased},
  author={Yu, Chao and Gao, Jiaxuan and Liu, Weilin and Xu, Botian and Tang, Hao and Yang, Jiaqi and Wang, Yu and Wu, Yi},
  journal={arXiv preprint arXiv:2302.01605},
  year={2023}
}

```
