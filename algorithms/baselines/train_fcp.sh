layout='cramped_room'

seed_max=12

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python sp_ppo.py --hidden_dim 128 --batch_size 4096 --mini_batch_size 128 --use_minibatch False --lr 9e-4 --gamma 0.99 \
    --epsilon 0.05 --entropy_coef 0.01 --layout ${layout} --num_episodes 3000 --seed ${seed}
    echo "seed${seed} finished !"
done

python FCP_stage1.py cramped_room

for seed in `seq 5`;
do
  python FCP_stage2.py --layout cramped_room --num_episodes 50000 --seed ${seed} --use_wandb --alt
#  python FCP_stage2.py --layout cramped_room --num_episodes 50000 --seed ${seed} --use_wandb
done
