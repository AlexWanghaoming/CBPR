layout='marshmallow_experiment'

python bpr_NN_scriptedPolicy.py --layout ${layout} --num_episodes 200 --mode intra --switch_human_freq 100 --Q_len 25 --seed 0
python bpr_NN_scriptedPolicy.py --layout ${layout} --num_episodes 200 --mode intra --switch_human_freq 100 --Q_len 50 --seed 0
python bpr_NN_scriptedPolicy.py --layout ${layout} --num_episodes 200 --mode intra --switch_human_freq 100 --Q_len 100 --seed 0
python bpr_NN_scriptedPolicy.py --layout ${layout} --num_episodes 200 --mode intra --switch_human_freq 100 --Q_len 200 --seed 0

#seed_max=5  # 跑seed_max个种子
#  for seed in `seq ${seed_max}`;
#  do
#    echo "seed is ${seed}:"
#    python bpr_NN_scriptedPolicy.py --layout ${layout} --num_episodes 200 --mode intra --switch_human_freq 100 --seed ${seed}
##    python bpr_RNN_skill_levels.py --layout ${layout} --num_episodes 50 --seed ${seed}
#
#    python okr_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --Q_len 25 --rho 0.5 --seed ${seed}
##    python okr_humanProxy.py --layout ${layout} --num_episodes 50 --Q_len 25 --rho 0.5 --seed ${seed}
#
#  done

