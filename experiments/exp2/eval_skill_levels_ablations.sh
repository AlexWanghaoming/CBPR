#layouts=('cramped_room' 'asymmetric_advantages' 'soup_coordination' 'coordination_ring')
layouts=('soup_coordination')

for layout in "${layouts[@]}"; do
  # 消融Q
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 10 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 10 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 10 --rho 0.1 --use_wandb
#
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 50 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 50 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 50 --rho 0.1 --use_wandb
#
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 5 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 5 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 5 --rho 0.1 --use_wandb

  # 消融rho
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.1 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.1 --use_wandb

#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.5 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.5 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.5 --use_wandb
#
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.9 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.9 --use_wandb
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.9 --use_wandb

### Ablations of number of meta-tasks
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.1
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.1
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.1
done

