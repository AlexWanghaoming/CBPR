#layouts=('cramped_room' 'asymmetric_advantages' 'marshmallow_experiment' 'coordination_ring')
layouts=('coordination_ring')

for layout in "${layouts[@]}"; do
#  python bpr_RNN_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 40
#  python bpr_RNN_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 40
#  python bpr_RNN_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 40

  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 10 --rho 0.5
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 10 --rho 0.5
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 10 --rho 0.5

#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --algorithm BCP
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --algorithm BCP
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --algorithm BCP
#
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --algorithm FCP
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --algorithm FCP
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --algorithm FCP
#
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --algorithm SP
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --algorithm SP
#  python evaluate_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --algorithm SP
done