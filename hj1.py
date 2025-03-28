import time

from metaworld import MT1

# from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy as p
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy as p

mt1 = MT1('pick-place-v2', seed=42)  # pick-place-v2 reach-v2
env = mt1.train_classes['pick-place-v2']()  # reach-v2
env.set_task(mt1.train_tasks[0])
obs, info = env.reset()

policy = p()

done = False

for i in range(100):
    print("i: ", i)
    # mt1 = MT1('reach-v2', seed=42)
    # env = mt1.train_classes['reach-v2']()
    # env.set_task(mt1.train_tasks[0])
    obs, info = env.reset()
    done = False
    while not done:
        a = policy.get_action(obs)
        obs, _, _, _, info = env.step(a)
        done = int(info['success']) == 1
        env.render()
        time.sleep(0.01)
