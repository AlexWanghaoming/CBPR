from utils import init_env
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src/')

"""
当前可用的scripted agents
SCRIPT_AGENTS = {
    "place_onion_in_pot":  functools.partial(SinglePeriodScriptAgent, period_name="pickup_onion_and_place_in_pot"),
    "place_tomato_in_pot":  functools.partial(SinglePeriodScriptAgent, period_name="pickup_tomato_and_place_in_pot"),
    "deliver_soup": functools.partial(SinglePeriodScriptAgent, period_name="pickup_soup_and_deliver"),
    "place_onion_and_deliver_soup": Place_Onion_and_Deliver_Soup_Agent,
    "place_tomato_and_deliver_soup": Place_Tomato_and_Deliver_Soup_Agent,
    # "noisy": Noisy_Agent,
    "put_onion_everywhere": functools.partial(SinglePeriodScriptAgent, period_name="put_onion_everywhere"),
    "put_tomato_everywhere": functools.partial(SinglePeriodScriptAgent, period_name="put_tomato_everywhere"),
    "put_dish_everywhere": functools.partial(SinglePeriodScriptAgent, period_name="put_dish_everywhere"),
    "random": StochasticScriptAgent,
    "pickup_tomato_and_place_mix": functools.partial(SinglePeriodScriptAgent, period_name="pickup_tomato_and_place_mix"),
    "pickup_ingredient_and_place_mix": functools.partial(SinglePeriodScriptAgent, period_name="pickup_ingredient_and_place_mix"),
    "mixed_order": functools.partial(SinglePeriodScriptAgent, period_name="mixed_order"),
}
"""


layout = 'cramped_room'
num_episodes = 50
env = init_env(layout=layout,
               agent0_policy_name='script:place_onion_in_pot',
               agent1_policy_name='script:deliver_soup',
               use_script_policy=True)

for k in range(1, num_episodes + 1):
    obs = env.reset()
    done = False
    episode_reward = 0
    agent_env_steps = 600 * (k - 1)
    while not done:
        obs_, sparse_reward, done, info = env.step((1, 1)) # if use scripted policy, just random an action here
        episode_reward += sparse_reward
        env.render(interval=0.01)
    print(f'Ep {k}:', episode_reward)