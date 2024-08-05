#!/bin/zsh

layouts=('cramped_room' 'coordination_ring' 'asymmetric_advantages')
#layouts=('soup_coordination')

seed_max=1

for layout in "${layouts[@]}"; do
  for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"

    python bcp.py --hidden_dim 128 --batch_size 4096 --lr 8e-4 --gamma 0.99 \
    --epsilon 0.05 --entropy_coef 0.01 --layout ${layout} --num_episodes 1500 \
    --seed ${seed} --use_wandb --alt

    echo "seed${seed} finished !"
  done
done

