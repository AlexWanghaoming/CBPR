#!/bin/zsh

#layouts=('cramped_room' 'asymmetric_advantages' 'marshmallow_experiment' 'coordination_ring')
layouts=('soup_coordination')

seed_max=5

for layout in "${layouts[@]}"; do
  for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"

    python bcp.py --hidden_dim 128 --batch_size 4096 --lr 8e-4 --gamma 0.99 \
    --epsilon 0.05 --entropy_coef 0.01 --layout ${layout} --num_episodes 2000 --seed ${seed}

    echo "seed${seed} finished !"
  done
done

