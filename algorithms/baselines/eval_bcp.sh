layouts=('cramped_room' 'asymmetric_advantages' 'marshmallow_experiment')

seed_max=5

for layout in "${layouts[@]}"; do
  for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"
#    python evaluate_bcp_humanProxy.py --layout ${layout} --num_episodes 50 --seed ${seed}
    python evaluate_bcp_scriptPolicy.py --layout ${layout} --num_episodes 200 --mode intra --switch_human_freq 100 --seed ${seed}
    echo "seed${seed} finished !"
  done
done

