#layouts=('cramped_room' 'asymmetric_advantages' 'coordination_ring' 'soup_coordination')
layouts=('asymmetric_advantages')
#layouts=('soup_coordination')

for layout in "${layouts[@]}"; do
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.1 --use_wandb

#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --algorithm BCP --use_wandb
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --algorithm BCP --use_wandb
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --algorithm BCP --use_wandb

#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --algorithm FCP --use_wandb
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --algorithm FCP --use_wandb
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --algorithm FCP --use_wandb
#
  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --algorithm SP
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --algorithm SP --use_wandb
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --algorithm SP --use_wandb
done