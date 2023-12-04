
#layouts=('cramped_room' 'asymmetric_advantages' 'marshmallow_experiment' 'coordination_ring')
layouts=('marshmallow_experiment')

seed_max=5

for layout in "${layouts[@]}"; do
  for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"
#    python bpr_RNN_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --Q_len 40
#    python bpr_RNN_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --Q_len 40
#    python bpr_RNN_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --Q_len 40
#    python bpr_RNN_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --Q_len 40
    python okr_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --Q_len 10 --rho 0.5
    python okr_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --Q_len 10 --rho 0.5
    python okr_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --Q_len 10 --rho 0.5
    python okr_scriptedPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --Q_len 10 --rho 0.5

#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --algorithm BCP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --algorithm BCP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --algorithm BCP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --algorithm BCP
#
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --algorithm FCP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --algorithm FCP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --algorithm FCP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --algorithm FCP
#
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --algorithm SP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --algorithm SP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --algorithm SP
#    python evaluate_scriptPolicy.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --algorithm SP
    echo "seed${seed} finished !"
  done
done

