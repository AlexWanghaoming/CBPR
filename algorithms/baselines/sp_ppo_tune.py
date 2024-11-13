import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from My_utils import Normalization, RewardScaling, ReplayBuffer
from agents.ppo_discrete import PPO_discrete
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../PPO-discrete/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from My_utils import seed_everything, LinearAnnealer, init_env
from ray import train, tune
from hyperopt import hp
import ray
from collections import deque
from ray.tune.search.hyperopt import HyperOptSearch


class FixedQueue(deque):
    def __init__(self, capacity):
        super().__init__(maxlen=capacity)  # 设置最大长度为队列容量
        self._capacity = capacity

    def append(self, item):
        if len(self) == self._capacity:
            # 如果队列已满，先弹出最左侧的元素（这是自动的，不需要手动操作，deque会自动处理）
            # 此行实际上是多余的，因为设置了maxlen，但是保留这里作为解释说明
            pass
        super().append(item)  # 添加元素到队列
        return self.average()  # 返回添加元素后的队列平均数

    def average(self):
        if not self:  # 如果队列为空，则返回0
            return 0
        return sum(self) / len(self)  # 计算队列中所有元素的平均数


gamma = 0.99
use_state_norm = False
use_reward_scaling = True
device = 'cpu'
num_episodes = 1000
hidden_dim = 128
layout = 'random3'
max_episode_steps =  600
state_dim = 96
action_dim = 6



def train_n_episodes(ego_agent:PPO_discrete, alt_agent:PPO_discrete, n_episodes:int, config:dict):
    queue = FixedQueue(10)
    annealer = LinearAnnealer(horizon=num_episodes * max_episode_steps * 0.5)
    env = init_env(layout=layout)
    ego_buffer = ReplayBuffer(4096, state_dim)
    alt_buffer = ReplayBuffer(4096, state_dim)
    cur_steps = 0  # Record the total steps during the training
    ego_state_norm = Normalization(shape=state_dim)
    alt_state_norm = Normalization(shape=state_dim)
    if use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=gamma)
    for k in range(1, n_episodes+1):
        agent_env_steps = max_episode_steps *  (k-1)
        reward_shaping_factor = annealer.param_value(agent_env_steps)

        obs  = env.reset()
        ego_obs, alt_obs = obs['both_agent_obs']
        if use_state_norm:
            ego_obs = ego_state_norm(ego_obs)
            alt_obs = alt_state_norm(alt_obs)
        if use_reward_scaling:
            reward_scaling.reset()

        episode_steps = 0
        done = False
        episode_reward = 0
        while not done:
            cur_steps += 1
            episode_steps += 1
            ego_a, ego_a_logprob = ego_agent.choose_action(ego_obs)
            alt_a, alt_a_logprob = alt_agent.choose_action(alt_obs)
            obs_, sparse_reward, done, info = env.step((ego_a, alt_a))
            shaped_r = info["shaped_r_by_agent"][0] + info["shaped_r_by_agent"][1]
            r = sparse_reward + shaped_r * reward_shaping_factor
            ego_obs_, alt_obs_ = obs_['both_agent_obs']
            episode_reward += r
            if use_state_norm:
                ego_obs_ = ego_state_norm(ego_obs_)
                alt_obs_ = ego_state_norm(alt_obs_)
            elif use_reward_scaling:
                reward = reward_scaling(r)
            if done:
                dw = True
            else:
                dw = False
            ego_buffer.store(ego_obs, ego_a, ego_a_logprob, r, ego_obs_, dw, done)
            alt_buffer.store(alt_obs, alt_a, alt_a_logprob, r, alt_obs_, dw, done)
            ego_obs = ego_obs_
            alt_obs = alt_obs_
            if ego_buffer.count == 4096:
                ego_agent.update(ego_buffer, cur_steps)
                alt_agent.update(alt_buffer, cur_steps)
                ego_buffer.count = 0
                alt_buffer.count = 0
        # print(f"Ep {k}:", episode_reward)
        queue.append(episode_reward)
        train.report({'mean_reward':queue.average()})


def run(config):
    test_env = init_env(layout=layout)
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n
    ego_agent = PPO_discrete(lr=config['lr'],
                         hidden_dim=hidden_dim,
                         batch_size=4096,
                         # batch_size=config['batch_size'],
                         epsilon=config['epsilon'],
                         entropy_coef=config['entropy_coef'],
                         state_dim=state_dim,
                         action_dim=action_dim,
                         num_episodes=num_episodes,
                         device=device)
    alt_agent = PPO_discrete(lr=config['lr'],
                         hidden_dim=hidden_dim,
                         batch_size=4096,
                         # batch_size=config['batch_size'],
                         epsilon=config['epsilon'],
                         entropy_coef=config['entropy_coef'],
                         state_dim=state_dim,
                         action_dim=action_dim,
                         num_episodes=num_episodes,
                         device=device)

    train_n_episodes(ego_agent=ego_agent, alt_agent=alt_agent, n_episodes=num_episodes, config=config)



if __name__ == '__main__':
    ray.init(num_cpus=4, local_mode=False)
    search_space = {
        "lr": hp.uniform('lr', 0.0005, 0.001),
        # "batch_size":hp.choice('batch_size',[2048, 4096]),
        'entropy_coef': hp.uniform('entropy_coef', 0.01, 0.1),
        'epsilon': hp.uniform('epsilon', 0.05, 0.2)

    }
    hyperopt_search = HyperOptSearch(search_space, metric="mean_reward", mode="max")
    analysis = tune.run(
        run,
        stop={'mean_reward': 250},
        metric="mean_reward",
        mode="max",
        name="overcooked_bayesopt",
        search_alg=hyperopt_search,
        num_samples=80,  # Total number of samples (trials) to evaluate
        resources_per_trial={"cpu": 1},  # Adjust the resources per trial
        verbose=1
    )
    
    # Get the best hyperparameter configuration and its performance
    best_config = analysis.get_best_config(metric="mean_reward", mode="max")
    print("Best config is:", best_config)


