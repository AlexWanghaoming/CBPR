
# 定义一个包含所有策略名称的数组
#declare -a scripted_policies=('place_onion_in_pot' 'deliver_soup' 'random' 'place_onion_and_deliver_soup')  # cramped_room
declare -a scripted_policies=('place_onion_in_pot' 'place_tomato_in_pot' 'deliver_soup' 'random' 'place_onion_and_deliver_soup' 'place_tomato_and_deliver_soup') # marshmallow_experiment

layout='marshmallow_experiment'

seed_max=1  # 跑seed_max个种子

for policy in "${scripted_policies[@]}"; do
    for seed in `seq ${seed_max}`;
    do
      echo "seed is ${seed}:"
      python mtp_scriptedPolicy.py --layout ${layout} --num_episodes 2000 --scripted_policy_name "${policy}" --seed ${seed}
    done
done

