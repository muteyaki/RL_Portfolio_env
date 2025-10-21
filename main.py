import os
from typing import Dict
from torch._C import device
from pg_basic_env_bak import PGBasicEnv
import logging
from stable_baselines3 import PPO, SAC, TD3, A2C
import json
import pandas as pd
from datetime import datetime
from feature_extract_backup import extract_layer
from stable_baselines3.common.env_util import make_vec_env

policy_map = {
    "ppo": PPO,
    "sac": SAC,
    "TD3": TD3,
    "a2c": A2C,
}
root_dir = "./exps"

SH50 = ['600196', '600276', '601088', '600340', '601111', '600547', '600585', '600690', '600837', '600887']
SZ = ['002020', '002025', '002032', '002033', '002041', '002069', '002095', '002074', '002077', '002091']

def evaluate(model, cfg):
    money_list = []
    daily_return  = []
    env = PGBasicEnv(cfg["trade_env"], SH50)
    obs = env.reset()
    done = False
    money_list.append(env.total_money)
    while not done:
        action, _ = model.predict(obs)
        next_obs, r, done, info = env.step(action)
        daily_return.append(info["money"]/money_list[-1]-1)
        money_list.append(info["money"])
        # print("current money is {}".format(info["money"]))
        obs = next_obs
    # cal return
    total_return = money_list[-1] / money_list[0] - 1
    daily_return=pd.DataFrame(daily_return)
    return daily_return,total_return


def rl_train(cfg: Dict,seed):
    date_time = datetime.now()
    date_time_str = date_time.strftime("%Y%m%d:%H%M%S")
    exp_dir = "{}/exp_{}".format(root_dir, date_time_str)
    #print(os.listdir("./"))
    os.mkdir(exp_dir)
    log_dir = "./log/{}_{}".format(cfg["agent"]["name"],cfg["agent"]["extractor"]["name"])
    #os.mkdir(log_dir)
    logging.basicConfig(filename = log_dir +"/{}_{}".format(seed,date_time_str)+'.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.DEBUG,filemode='a',datefmt='%Y-%m-%d%I:%M:%S %p')
    env = PGBasicEnv(cfg["train_env"],SH50)
    #env = make_vec_env(PGBasicEnv, n_envs=4, env_kwargs={"cfg": cfg["train_env"], "tic_list": SH50})
    # save config
    with open("{}/config.json".format(exp_dir), "w") as f:
        json.dump(cfg, f)
    # logging.basicConfig("")
    policy_kwargs = dict(
        features_extractor_class=extract_layer[cfg["agent"]["extractor"]["name"]],
        features_extractor_kwargs=dict(cfg = cfg["agent"]["extractor"]))
    model_type = policy_map[cfg["agent"]["name"]]
    model = model_type("MultiInputPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1, seed=seed,device="cuda")
    _,total_return = evaluate(model, cfg)
    logging.info("the seed is {}".format(seed))
    print("init policy total return is {}".format(total_return))
    logging.info("init return money is {}".format(total_return))
    max_return = total_return
    for i in range(6):
        model.learn(10000)
        daily_return,current_return = evaluate(model, cfg)
        print("current epoch is {}, current return is {}".format(i, current_return))
        if current_return > max_return:
            model.save("{}/best_policy".format(exp_dir))
            max_return = current_return
            daily_return.to_csv('{}_{}_{}_daily_return_SH.csv'.format(cfg["agent"]["name"],cfg["agent"]["extractor"]["name"],seed))
    logging.info("max return money is {}".format(max_return))
    print("max return is {}".format(max_return))

if __name__ == "__main__":
    with open("config/cfg_train.json", "r") as f:
        config = json.load(f)
    for i in range(10):
        rl_train(config,50+i)






