"""
Sample collection implementation through interaction between agent and environment
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from envs.multi_task_game_avoid import MultiTaskGameAvoid

from pearl.algorithm.sac import SAC


class Sampler:
    """Data sampling class"""
    def __init__(
        self,
        env: MultiTaskGameAvoid,
        agent: SAC,
        max_step: int,
        device: torch.device,
    ) -> None:

        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.device = device

    def obtain_samples(
        self,
        task_index: int, 
        accum_context: bool = True,
    ) -> Tuple[List[Dict[str, np.ndarray]], int]:
        """
        Rollout one trajectory to obtain samples up to the number of maximum samples
        param:
            `max_samples` the maximum of samples
            `update_posterior` whether update posterior
            `accum_context` whether accumulate context or not
        return:
            `trajs: List[Dict[str, np.ndarray]]` a list of traj
            `cur_samples` num of samples
        """
        trajs = []
        cur_samples = 0

        traj = self.rollout(task_index=task_index, accum_context=accum_context)
        trajs.append(traj)
        cur_samples += len(traj["cur_obs"])
        self.agent.encoder.sample_z()

        return trajs, cur_samples

    def rollout(self,
        task_index: int,
        accum_context: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Rollout up to maximum trajectory length or done to yield transition
        param:
            `accum_context` whether accumulate context or not
        return:
            transition dict {s, a, ,r, s', d, info}
        """
        _cur_obs = []
        _actions = []
        _rewards = []
        _next_obs = []
        _dones = []


        obs = self.env.reset_task(task_index)
        done = False
        cur_step = 0

        while not (done or cur_step == self.max_step):
            action = self.agent.get_action(obs)
            next_obs, reward, terminated, truncated = self.env.step(action[0], action[1])

            done = terminated or truncated

            # Update the agent's current context
            if accum_context:
                self.update_context(obs=obs, action=action, reward=reward)

            _cur_obs.append(obs)
            _actions.append(action)
            _rewards.append(reward)
            _next_obs.append(next_obs)
            _dones.append(done)

            cur_step += 1
            obs = next_obs
        return dict(
            cur_obs=np.array(_cur_obs),
            actions=np.array(_actions),
            rewards=np.array(_rewards).reshape(-1, 1),
            next_obs=np.array(_next_obs),
            dones=np.array(_dones).reshape(-1, 1),

        )

    def update_context(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray) -> None:
        """
        Append single transition to the current context
        context is like (1, batch_size, s+a+r)
        param:
            `obs` current observation
            `action` current action
            `reward` current reward
        """
        obs = obs.reshape((1, 1, *obs.shape))
        action = action.reshape((1, 1, *action.shape))
        reward = reward.reshape((1, 1, *reward.shape))

        obs = torch.from_numpy(obs).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        
        # for test
        # print('obs shape is {}'.format(obs.shape))
        # print('action shape is {}'.format(action.shape))
        # print('reward shape is {}'.format(reward.shape))
        transition = torch.cat([obs, action, reward], dim=-1).to(self.device)

        if self.agent.encoder.context is None:
            self.agent.encoder.context = transition
        else:
            # accumulate context as (1, batch_size, s+a+r)
            self.agent.encoder.context = torch.cat([self.agent.encoder.context, transition], dim=1).to(
                self.device,
            )
