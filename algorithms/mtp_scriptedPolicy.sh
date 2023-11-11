
# 定义一个包含所有策略名称的数组
declare -a scripted_policies=('place_onion_in_pot' 'deliver_soup' 'random' 'place_onion_and_deliver_soup')

layout='cramped_room'

seed_max=1  # 跑多个种子

for policy in "${scripted_policies[@]}"; do
    for seed in `seq ${seed_max}`;
    do
      echo "seed is ${seed}:"
      python mtp_scriptedPolicy.py --layout ${layout} --num_episodes 2000 --scripted_policy_name "${policy}" --seed ${seed}
    done
done

