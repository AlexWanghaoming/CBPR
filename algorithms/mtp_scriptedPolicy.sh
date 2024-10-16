
# 定义一个包含所有策略名称的数组
#declare -a scripted_policies=('place_onion_in_pot' 'deliver_soup' 'random' 'place_onion_and_deliver_soup')  # cramped_room
#declare -a scripted_policies=('place_onion_in_pot' 'place_tomato_in_pot' 'deliver_soup' 'random' 'place_onion_and_deliver_soup' 'place_tomato_and_deliver_soup') # marshmallow_experiment
#declare -a scripted_policies=('place_onion_in_pot' 'deliver_soup' 'random' 'place_onion_and_deliver_soup')  # asymmetric_advantages
#declare -a scripted_policies=('place_onion_in_pot' 'place_tomato_in_pot' 'deliver_soup' 'random' 'put_onion_everywhere' 'put_tomato_everywhere')  # counter_circuit
#declare -a scripted_policies=('place_onion_in_pot' 'deliver_soup' 'place_onion_and_deliver_soup' 'random')  # coordination_ring
#declare -a scripted_policies=('place_onion_in_pot' 'deliver_soup' 'put_onion_everywhere' 'put_dish_everywhere' 'random')  # random3
#declare -a scripted_policies=('place_tomato_in_pot' 'deliver_soup' 'mixed_order' 'random')  # soup_coordination
declare -a scripted_policies=('place_onion_in_pot' 'place_onion_and_deliver_soup')  # soup_coordination
#declare -a scripted_policies=('place_onion_in_pot' 'place_onion_and_deliver_soup' 'place_tomato_and_deliver_soup' 'pickup_tomato_and_place_mix' 'pickup_ingredient_and_place_mix')  # soup_coordination

layout='soup_coordination'

seed_max=1  # 跑seed_max个种子

for policy in "${scripted_policies[@]}"; do
    for seed in `seq ${seed_max}`;
    do
      echo "seed is ${seed}:"
      python mtp_scriptedPolicy.py --layout ${layout} --num_episodes 5000 --scripted_policy_name "${policy}" --seed ${seed}
    done
done

