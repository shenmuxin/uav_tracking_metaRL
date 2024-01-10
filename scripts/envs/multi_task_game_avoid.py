#! /usr/bin/env python
# coding :utf-8
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from envs.game_avoid import GameAvoid

import rospy

import numpy as np
from typing import List, Dict, Any


class MultiTaskGameAvoid(GameAvoid):

    def __init__(self, num_tasks: int , **kwargs):
        super().__init__(**kwargs)
        self.tasks: List[Dict[str, Any]] = self.sample_tasks(num_tasks)
        # current task
        self._task = self.tasks[0]
        
        # take off flag
        self.start_finish = False

        # meta gradient flag
        self.meta_gradient_flag = False

    
    def sample_tasks(self, num_tasks: int):
        np.random.seed(0)
        # specify vx and yaw_rate range
        vxs = np.linspace(start=0.6, stop=0.7, num=num_tasks)
        yaw_rates = np.linspace(start=-0.04, stop=0.04, num=num_tasks)
        tasks = [{'vx': vx, 'yaw_rate': yaw_rate} for vx, yaw_rate in zip(vxs, yaw_rates)]
        return tasks
    
    def reset_task(self, idx: int):
        # change to task idx
        self._task = self.tasks[idx]
        print("Current task index is {} vx:{} yaw_rate:{}".format(idx, self._task['vx'], self._task['yaw_rate']))
        if self.start_finish:
            # fly home
            obs = self.reset()
        if not self.start_finish:
            # first take off
            obs = self.start()
            self.start_finish = True
            
        return obs


    def get_all_task_idx(self):
        return list(range(len(self.tasks)))
    
    def _avoiderCtrl(self, event):
        """
        func->
            control avoider flight pattern
        """
        if self.flight_flag == True:
            if self.meta_gradient_flag == False:
                # change to current task
                self.avoider_ctl.moveByVelocityYawrateFLU(vx=self._task['vx'], vy=0, vz=0, yaw_rate=self._task['yaw_rate'])
            if self.meta_gradient_flag == True:
                # stop avoider in place
                self.avoider_ctl.moveByVelocityYawrateFLU(vx=0, vy=0, vz=0, yaw_rate=0)

    def begin_meta_gradient(self, flag: bool):
        self.meta_gradient_flag = flag

        


if __name__ == "__main__":

    rospy.init_node("test_multi_task_game_node")

    # initialize env
    env = MultiTaskGameAvoid(num_tasks=3)
    # initialize tasks
    tasks: List[int] = env.get_all_task_idx()

    trajectories_list = []

    start_finish = False
    for idx in tasks:
        observation = env.reset_task(idx)
        print("initial shape is {}".format(observation.shape))

        done = False
        # begin to record traj
        env.set_trajectory_record(True)
        while not done:
            observation_, reward, terminated, truncated = env.step(vx=0.02, yaw_rate=0)
            print("step shape is {}".format(observation_.shape))
            print('reward is {}, shape is {}'.format(reward, reward.shape))
            done = terminated or truncated
            observation = observation_
        # end to record traj
        print(env.step_count)
        env.set_trajectory_record(False)

        trajectories_list.append(env.get_trajectory())
    # save
    np.save('/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/utils/results' + '/trajectories_list.npy', np.array(trajectories_list))
    # plot
    import matplotlib.pyplot as plt
    for trajectories in trajectories_list:
        avoider_trajs = trajectories[0]
        plt.plot(avoider_trajs[:, 0], avoider_trajs[:, 1], label="avoider")
    plt.legend()
    plt.show()
