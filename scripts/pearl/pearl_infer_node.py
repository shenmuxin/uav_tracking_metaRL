#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

print('\n*****************************************\n\t[test libraries]:\n')
import rospy
print(' - rospy.__file__ = %s'%rospy.__file__)


import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(parent_path)
# activate conda env
sys.path.append('/home/sjh/anaconda3/envs/metarl/lib/python3.8/site-packages')
import inspect

from envs.multi_task_game_avoid import MultiTaskGameAvoid
from pearl.algorithm.meta_learner import MetaLearner
from pearl.algorithm.sac import SAC
print(' - SAC.__file__ = %s'%inspect.getfile(SAC))

from typing import Any, Dict, List
import numpy as np
import torch
import yaml
print('\n*****************************************\n\t[finish test]\n')

if __name__ == "__main__":
    # register node
    rospy.init_node('pearl_trainer_node')

    # 加载实验环境设置的超参数
    with open(os.path.join(current_path, "configs/inference_config.yaml"), "r", encoding="utf-8") as file:
        experiment_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # 加载超参数以设置目标奖励
    with open(
        os.path.join(current_path, "configs/" + experiment_config["env_name"] + "_target_config.yaml"),
        "r",
        encoding="utf-8",
    ) as file:
        env_target_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # 创建多任务环境和示例任务
    env: MultiTaskGameAvoid = MultiTaskGameAvoid(num_tasks=env_target_config["train_tasks"],)
    tasks: List[int] = env.get_all_task_idx()

    # 设置随机种子值
    np.random.seed(experiment_config["seed"])
    torch.manual_seed(experiment_config["seed"])

    observ_dim: int = 47
    action_dim: int = 2
    hidden_dim: int = env_target_config["hidden_dim"]

    device: torch.device = (
        torch.device("cuda", index=experiment_config["gpu_index"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    agent = SAC(
        observ_dim=observ_dim,
        action_dim=action_dim,
        latent_dim=env_target_config["latent_dim"],
        hidden_dim=hidden_dim,
        encoder_input_dim=observ_dim + action_dim + 1,
        encoder_output_dim=env_target_config["latent_dim"] * 2,
        device=device,
        **env_target_config["sac_params"],
    )

    meta_learner = MetaLearner(
        env=env,
        agent=agent,
        observ_dim=observ_dim,
        action_dim=action_dim,
        train_tasks=tasks,
        test_tasks=tasks,
        save_exp_name=experiment_config["save_exp_name"],
        save_file_name=experiment_config["save_file_name"],
        load_exp_name=experiment_config["load_exp_name"],
        load_file_name=experiment_config["load_file_name"],
        load_ckpt_num=experiment_config["load_ckpt_num"],
        device=device,
        **env_target_config["pearl_params"],
    )

    # begin inference
    trajectories_list = []
    for idx in tasks:
        meta_learner.inference(idx, 2, trajectories_list)
    print(np.array(trajectories_list).shape)