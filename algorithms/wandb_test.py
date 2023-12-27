import wandb
import os
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
wandb.init(project='overcooked_rl',
               group='MTP',
               name=f'tt',
               job_type='training',
               reinit=True)