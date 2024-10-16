import wandb
import glob
import os, sys

LAYOUT = sys.argv[1]
if not LAYOUT:
    LAYOUT = 'cramper_room'

SOURCE_DIR = 'path/to/CBPR'
WANDB_PATH = SOURCE_DIR + '/algorithms/baselines/wandb'
POLICY_POOL_PATH = SOURCE_DIR + "/models/policy_pool"

runs = glob.glob(f"{WANDB_PATH}/run*")
run_ids = [x.split('-')[-1] for x in runs]
print(runs)
print(run_ids)
api = wandb.Api()
i = 0

fcp_s1_dir = f"{POLICY_POOL_PATH}/{LAYOUT}/fcp/s1"
if not os.path.exists(fcp_s1_dir):
    os.makedirs(fcp_s1_dir, exist_ok=True)  # 创建目录，如果目录已经存在，则不会引发异常

for run_id in run_ids:
    run = api.run(f"wanghm/overcooked_rl/{run_id}")
    if run.state == "finished" and run.group == 'FCP' and LAYOUT in run.name:
        i += 1
        final_ep_sparse_r = run.summary['ep_reward']
        history = run.history()
        history = history[['_step', 'ep_reward']]
        steps = history['_step'].to_numpy()
        ep_sparse_r = history['ep_reward'].to_numpy()
        files = run.files()
        actor_pts = [f for f in files if f.name.startswith('sp_periodic')]
        actor_versions = [eval(f.name.split('_')[-1].split('.pt')[0]) for f in actor_pts]
        actor_pts = {v: p for v, p in zip(actor_versions, actor_pts)}
        actor_versions = sorted(actor_versions)
        max_actor_versions = max(actor_versions) + 1
        max_steps = max(steps)
        new_steps = [steps[0]]
        new_ep_sparse_r = [ep_sparse_r[0]]
        for s, er in zip(steps[1:], ep_sparse_r[1:]):
            l_s = new_steps[-1]
            l_er = new_ep_sparse_r[-1]
            for w in range(l_s + 1, s, 100):
                new_steps.append(w)
                new_ep_sparse_r.append(l_er + (er - l_er) * (w - l_s) / (s - l_s))
        steps = new_steps
        ep_sparse_r = new_ep_sparse_r

        # select checkpoints
        selected_pts = dict(init=0, mid=-1, final=max_steps)
        mid_ep_sparse_r = final_ep_sparse_r / 2
        min_delta = 1e9
        for s, score in zip(steps, ep_sparse_r):
            if min_delta > abs(mid_ep_sparse_r - score):
                min_delta = abs(mid_ep_sparse_r - score)
                selected_pts["mid"] = s
        selected_pts = {k: int(v / max_steps * max_actor_versions) for k, v in selected_pts.items()}
        for tag, exp_version in selected_pts.items():
            version = actor_versions[0]
            for actor_version in actor_versions:
                if abs(exp_version - version) > abs(exp_version - actor_version):
                    version = actor_version
            print(f"sp{i}", tag, "Expected", exp_version, "Found", version)
            ckpt = actor_pts[version]
            ckpt.download("tmp", replace=True)  # 在当前目录新建一个tmp目录并将选择的checkpoint下载进取
            os.system(f"mv tmp/sp_periodic_{version}.pt {fcp_s1_dir}/sp{i}_{tag}_actor.pt")