from human_aware_rl.rllib.rllib import load_agent
import pickle
with open('/alpha/overcooked_rl/src/overcooked_demo/server/static/assets/agents/RandAI/agent.pickle', 'rb') as f1:
    data = pickle.load(f1)
print(data)