import pickle
import os, sys

HUMAN_TEST_DIR = '../../src/overcooked_demo/real_human_test/marshmallow_experiment/Old/AH/qqqqqqqqqqqqqqq/2023-11-29_13-00-51/result.pkl'

# 以二进制读取模式打开文件
with open(HUMAN_TEST_DIR, 'rb') as file:
    # 加载pickle文件
    data = pickle.load(file)

# 使用加载的数据
print(data['final_reward'])