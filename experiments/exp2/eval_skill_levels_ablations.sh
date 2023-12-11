layouts=('cramped_room' 'asymmetric_advantages' 'marshmallow_experiment' 'coordination_ring')
#layouts=('cramped_room')

for layout in "${layouts[@]}"; do
  # 消融Q
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 10 --rho 0.1
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 10 --rho 0.1
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 10 --rho 0.1
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 50 --rho 0.1
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 50 --rho 0.1
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 50 --rho 0.1
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.1
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.1
#  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.1
  # 消融rho
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.1
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.1
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.1

  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.5
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.5
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.5

  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level high --Q_len 20 --rho 0.9
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level medium --Q_len 20 --rho 0.9
  python okr_skill_levels.py --layout ${layout} --num_episodes 50 --skill_level low --Q_len 20 --rho 0.9
done