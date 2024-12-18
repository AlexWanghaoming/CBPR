
layouts=('cramped_room' 'asymmetric_advantages' 'coordination_ring')
#layouts=('marshmallow_experiment')
#layouts=('soup_coordination')

seed_max=5

for layout in "${layouts[@]}"; do
  for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --algorithm BCP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --algorithm BCP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --algorithm BCP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --algorithm BCP

#    python okr_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --Q_len 5 --rho 0.1
#    python okr_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --Q_len 20 --rho 0.1
    python okr_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --Q_len 5 --rho 0.1
    python okr_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --Q_len 5 --rho 0.1
    python okr_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --Q_len 5 --rho 0.1

    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --algorithm FCP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --algorithm FCP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --algorithm FCP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --algorithm FCP

    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 100 --seed ${seed} --algorithm SP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode intra --switch_human_freq 200 --seed ${seed} --algorithm SP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 1 --seed ${seed} --algorithm SP
    python evaluate_alter_skill_levels.py --layout ${layout} --num_episodes 50 --mode inter --switch_human_freq 2 --seed ${seed} --algorithm SP
    echo "seed${seed} finished !"
  done
done

