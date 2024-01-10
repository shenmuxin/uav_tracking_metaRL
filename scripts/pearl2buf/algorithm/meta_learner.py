import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(parent_path)


import datetime
import time
import warnings
from collections import deque
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import torch


from envs.multi_task_game_avoid import MultiTaskGameAvoid
import tensorboardX
print('- tensorboardX.__file__ = %s'%tensorboardX.__file__)

from tqdm import tqdm

from pearl.algorithm.buffers import MultiTaskReplayBuffer
from pearl.algorithm.sac import SAC
from pearl.algorithm.sampler import Sampler


class MetaLearner:
    def __init__(
        self,
        env: MultiTaskGameAvoid,
        agent: SAC,
        observ_dim: int,
        action_dim: int,
        train_tasks: List[int],
        test_tasks: List[int],
        save_exp_name: str,
        save_file_name: str,
        load_exp_name: str,
        load_file_name: str,
        load_ckpt_num: int,
        device: torch.device,
        **config,
    ) -> None:
        self.env = env

        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.device = device

        self.num_iterations: int = config["num_iterations"]
        self.num_sample_tasks: int = config["num_sample_tasks"]

        self.num_init_trajs: int = config["num_init_trajs"]
        self.num_prior_trajs: int = config["num_prior_trajs"]
        self.num_posterior_trajs: int = config["num_posterior_trajs"]

        self.num_meta_grads: int = config["num_meta_grads"]
        self.meta_batch_size: int = config["meta_batch_size"]
        self.batch_size: int = config["batch_size"]
        self.max_step: int = config["max_step"]

        self.sampler = Sampler(env=env, agent=agent, max_step=config["max_step"], device=device)

        # detach RL buffer and Encoder buffer
        # - RL replay buffer
        # - encoder replay buffer
        self.rl_replay_buffer = MultiTaskReplayBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )
        self.encoder_replay_buffer = MultiTaskReplayBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )

        # save setting
        if not save_file_name:
            save_file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.result_path = os.path.join("/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/logs", save_exp_name, save_file_name)
        if save_exp_name == "train":
            self.writer = tensorboardX.SummaryWriter(log_dir=self.result_path)
        elif save_exp_name == "test":
            os.makedirs(self.result_path, exist_ok=True)

        # load setting
        if load_exp_name and load_file_name:
            ckpt_path = os.path.join(
                "/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/logs",
                load_exp_name,
                load_file_name,
                "checkpoint_" + str(load_ckpt_num) + ".pt",
            )
            ckpt = torch.load(ckpt_path)

            self.agent.policy.load_state_dict(ckpt["policy"])
            self.agent.encoder.load_state_dict(ckpt["encoder"])
            self.agent.qf1.load_state_dict(ckpt["qf1"])
            self.agent.qf2.load_state_dict(ckpt["qf2"])
            self.agent.target_qf1.load_state_dict(ckpt["target_qf1"])
            self.agent.target_qf2.load_state_dict(ckpt["target_qf2"])
            self.agent.log_alpha = ckpt["log_alpha"]
            self.agent.alpha = ckpt["alpha"]
            self.rl_replay_buffer = ckpt["rl_replay_buffer"]
            self.encoder_replay_buffer = ckpt["encoder_replay_buffer"]

        # set early stopping conditions
        self.dq: deque = deque(maxlen=config["num_stop_conditions"])        # deque to saving early stopping conditions
        self.num_stop_conditions: int = config["num_stop_conditions"]       # num of early stopping conditions
        self.stop_goal: int = config["stop_goal"]                           # the goal return of stop
        self.is_early_stopping = False                                      # whether is stop or not

        # record saving
        self.return_before_infer = []
        self.return_after_infer = []

    def sample_context(self, indices: np.ndarray) -> torch.Tensor:
        """
        For given task index, sample context in encoder replay buffer
        params:
            `indices` is the np.ndarray of task index shape is (1, meta_batch_size)
        return:
            context is defined as {s, a, r, s'} in paper
            but context is defined as {s, a, r} here
            context batch likes
                    [[[s,a,r]
                    [s,a,r]
                    ...]]
            shape of context batch is (meta_batch_size, batch_size, s+a+r)
        """
        
        context_batch = []
        for index in indices:
            batch = self.encoder_replay_buffer.sample_batch(task=index, batch_size=self.batch_size)
            context_batch.append(
                np.concatenate((batch["cur_obs"], batch["actions"], batch["rewards"]), axis=-1),
            )
        return torch.Tensor(context_batch).to(self.device)

    def sample_transition(self, indices: np.ndarray) -> List[torch.Tensor]:
        """
        For given task index, sample transition in rl replay buffer
        params:
            `indices` is the np.ndarray of task index
        return:
            transition is defined as {s, a, r, s', d} in rl manner
            each element shape is (meta_batch_size, batch_size, feature)
        """
        cur_obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for index in indices:
            batch = self.rl_replay_buffer.sample_batch(task=index, batch_size=self.batch_size)
            cur_obs.append(batch["cur_obs"])
            actions.append(batch["actions"])
            rewards.append(batch["rewards"])
            next_obs.append(batch["next_obs"])
            dones.append(batch["dones"])

        cur_obs = torch.Tensor(cur_obs).view(len(indices), self.batch_size, -1).to(self.device)
        actions = torch.Tensor(actions).view(len(indices), self.batch_size, -1).to(self.device)
        rewards = torch.Tensor(rewards).view(len(indices), self.batch_size, -1).to(self.device)
        next_obs = torch.Tensor(next_obs).view(len(indices), self.batch_size, -1).to(self.device)
        dones = torch.Tensor(dones).view(len(indices), self.batch_size, -1).to(self.device)
        return [cur_obs, actions, rewards, next_obs, dones]

    def collect_train_data(
        self,
        task_index: int,
        traj_num: int,
        update_posterior: bool,
        add_to_enc_buffer: bool,
    ) -> None:
        """
        Rollout to collect the trajs data of given task index for rl buffer and encoder buffer
        params:
            `task_index` is the index of train task
            `traj_num` is the num of trajectory need to rollout
            `update_posterior` whether update z ~ posterior q(z|c) or not
            `add_to_enc_buffer` whether add train data to encoder buffer
        """
        self.agent.encoder.clear_z()
        self.agent.policy.is_deterministic = False

        cur_samples = 0
        for i in range(traj_num):
            # rollout to get train samples
            trajs, num_samples = self.sampler.obtain_samples(
                task_index=task_index,
                accum_context=False,
            )
            cur_samples += num_samples

            # add trajs data to rl replay buffer
            self.rl_replay_buffer.add_trajs(task_index, trajs)
            if add_to_enc_buffer:
                # add trajs data to encoder replay buffer
                self.encoder_replay_buffer.add_trajs(task_index, trajs)

            if update_posterior:
                # 根据采样上下文进行事后更新
                context_batch = self.sample_context(np.array([task_index]))     # context_batch shape is (meta_batch_size, batch_size, s+a+r)
                                                                                # (1, 256, 27) during sampling, (4, 256, 27) during meta-gradient
                self.agent.encoder.infer_posterior(context_batch)

    def meta_train(self) -> None:
        # 元训练
        total_start_time: float = time.time()
        for iteration in range(self.num_iterations):
            start_time: float = time.time()

            # only for first meta_iteration，all meta-train task trajs are stored 
            # into both rl buffer and encoder buffer util buffer satisfy initial size of pool
            if iteration == 0:
                print("Collecting initial pool of data for train and eval")
                # for each task collect 5 trajs
                for index in tqdm(self.train_tasks):
                    self.env.reset_task(index)
                    self.collect_train_data(
                        task_index=index,
                        traj_num=self.num_init_trajs,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

                # save tree map
                tree_map = self.env.get_map()
                np.save(self.result_path + '/tree_map_list.npy', tree_map)

            print(f"\n=============== Iteration {iteration} ===============")

            # for any task store the new trajs into corresponding buffer
            for i in range(self.num_sample_tasks):
                # random choose a task
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.encoder_replay_buffer.task_buffers[index].clear()

                # 收集样本 z ~ prior r(z) 的先验分布trajs
                # if self.num_prior_samples > 0:
                if self.num_prior_trajs > 0:
                    print(f"[{i + 1}/{self.num_sample_tasks}] collecting samples with prior")
                    self.collect_train_data(
                        task_index=index,
                        traj_num=self.num_prior_trajs,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

                # 使用先验 z ~ prior r(z) 生成的路径数据来训练编码器
                # 使用后验 z ~ posterior q(z|c) 生成的路径数据来训练RL policy
                # if self.num_posterior_samples > 0:
                if self.num_posterior_trajs > 0:
                    print(f"[{i + 1}/{self.num_sample_tasks}] collecting samples with posterior")
                    self.collect_train_data(
                        task_index=index,
                        traj_num=self.num_posterior_trajs,
                        update_posterior=True,
                        add_to_enc_buffer=False,
                    )

            # use meta_bs tasks sampled to update networks
            print(f"Start meta-gradient updates of iteration {iteration}")
            self.env.begin_meta_gradient(True)

            for i in range(self.num_meta_grads):
                # random choose task index up to meta_batch_size shape is (1, meta_batch_size)
                indices: np.ndarray = np.random.choice(self.train_tasks, self.meta_batch_size)

                # initialize encoder context
                self.agent.encoder.clear_z(num_tasks=len(indices))

                # Context 采样
                context_batch: torch.Tensor = self.sample_context(indices)

                # RL batches 采样
                transition_batch: List[torch.Tensor] = self.sample_transition(indices)

                # 从 SAC 算法学习策略、Q 函数和编码器网络
                log_values: Dict[str, float] = self.agent.train_model(
                    meta_batch_size=self.meta_batch_size,
                    batch_size=self.batch_size,
                    context_batch=context_batch,
                    transition_batch=transition_batch,
                )

                # 阻止编码器任务变量 z 的反向传播 Back Propagation
                self.agent.encoder.task_z.detach()
            self.env.begin_meta_gradient(False)

            # 元测试任务中学习表现的评估
            print(f"Start meta-test of iteration {iteration}")
            self.meta_test(iteration, total_start_time, start_time, log_values)

            # if self.is_early_stopping:
            #     print(
            #         f"\n==================================================\n"
            #         f"The last {self.num_stop_conditions} meta-testing results are {self.dq}.\n"
            #         f"And early stopping condition is {self.is_early_stopping}.\n"
            #         f"Therefore, meta-training is terminated.",
            #     )
            #     break
            
            # period saving
            if (iteration + 1) % 50 == 0:
                ckpt_path = os.path.join(self.result_path, "checkpoint_" + str(iteration + 1) + ".pt")
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                        "encoder": self.agent.encoder.state_dict(),
                        "qf1": self.agent.qf1.state_dict(),
                        "qf2": self.agent.qf2.state_dict(),
                        "target_qf1": self.agent.target_qf1.state_dict(),
                        "target_qf2": self.agent.target_qf2.state_dict(),
                        "log_alpha": self.agent.log_alpha,
                        "alpha": self.agent.alpha,
                        "rl_replay_buffer": self.rl_replay_buffer,
                        "encoder_replay_buffer": self.encoder_replay_buffer,
                    },
                    ckpt_path,
                )

    def collect_test_data(
        self,
        task_index: int,
        traj_num: int,
        update_posterior: bool,
    ) -> List[List[Dict[str, np.ndarray]]]:
        """
        Rollout to collect the trajs data of given task index for rl buffer and encoder buffer
        params:
            `task_index` is the index of test task
            `traj_num` is the num of trajectory need to rollout
            `update_posterior` whether update z ~ posterior q(z|c) or not
        """
        # 收集元测试任务的路径数据
        self.agent.encoder.clear_z()
        # TODO: figure out why here policy need to set as deterministic
        self.agent.policy.is_deterministic = True       # why the policy is deterministic

        cur_trajs = []
        cur_samples = 0
        for i in range(traj_num):
            trajs, num_samples = self.sampler.obtain_samples(
                task_index=task_index,
                accum_context=True,
            )
            cur_trajs.append(trajs)
            cur_samples += num_samples

            # 根据 posterior 后验分布来更新 context
            self.agent.encoder.infer_posterior(self.agent.encoder.context)
        return cur_trajs


    def meta_test(
        self,
        iteration: int,
        total_start_time: float,
        start_time: float,
        log_values: Dict[str, float],
    ) -> None:
        # 元测试
        test_results = {}
        return_before_infer = 0
        return_after_infer = 0
        run_cost_before_infer = np.zeros(self.max_step)
        run_cost_after_infer = np.zeros(self.max_step)

        # test all self.test_tasks is too long time
        # for index in self.test_tasks:

        # uniformly choose a task for test
        index = np.random.choice(self.test_tasks)
        self.env.reset_task(index)
        trajs: List[List[Dict[str, np.ndarray]]] = self.collect_test_data(
            task_index=index,
            traj_num=2,
            update_posterior=True,
        )

        return_before_infer += np.sum(trajs[0][0]["rewards"])
        return_after_infer += np.sum(trajs[1][0]["rewards"])
            

        test_results["return_before_infer"] = return_before_infer / len(self.test_tasks)
        test_results["return_after_infer"] = return_after_infer / len(self.test_tasks)

        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["qf1_loss"] = log_values["qf1_loss"]
        test_results["qf2_loss"] = log_values["qf2_loss"]
        test_results["encoder_loss"] = log_values["encoder_loss"]
        test_results["alpha_loss"] = log_values["alpha_loss"]
        test_results["alpha"] = log_values["alpha"]
        test_results["z_mean"] = log_values["z_mean"]
        test_results["z_var"] = log_values["z_var"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        self.visualize_within_tensorboard(test_results, iteration)

        # early stopping, if continuously satisfy for num_stop_conditions iteration
        # self.dq.append(test_results["return_after_infer"])
        # if all(list(map((lambda x: x >= self.stop_goal), self.dq))):
        #     self.is_early_stopping = True

        # save model
        # if self.is_early_stopping:
        #     ckpt_path = os.path.join(self.result_path, "checkpoint_" + str(iteration) + ".pt")
        #     torch.save(
        #         {
        #             "policy": self.agent.policy.state_dict(),
        #             "encoder": self.agent.encoder.state_dict(),
        #             "qf1": self.agent.qf1.state_dict(),
        #             "qf2": self.agent.qf2.state_dict(),
        #             "target_qf1": self.agent.target_qf1.state_dict(),
        #             "target_qf2": self.agent.target_qf2.state_dict(),
        #             "log_alpha": self.agent.log_alpha,
        #             "alpha": self.agent.alpha,
        #             "rl_replay_buffer": self.rl_replay_buffer,
        #             "encoder_replay_buffer": self.encoder_replay_buffer,
        #         },
        #         ckpt_path,
        #     )


    def inference(
        self,
        task_index: int,
        traj_num: int,
        trajectories_list: List[Any],
    ) -> None:
        """
        Use trained model to inference within specific task, to rollout `traj_num` trajectories.
        params:
            `task_index` is the index of task need to inference
            `traj_num` is the num of trajectory need to rollout
        """

        # sample z ~ r(z) prior
        self.agent.encoder.clear_z()
        # TODO: figure out why here policy need to set as deterministic
        self.agent.policy.is_deterministic = True       # why the policy is deterministic

        self.env.get_map()
        for i in range(traj_num):
            obs = self.env.reset_task(task_index)
            done = False
            cur_step = 0

            # save tree map
            if i == 0:
                tree_map = self.env.get_map()
                np.save(self.result_path + '/tree_map_list.npy', tree_map)

            # begin to record trajectory
            self.env.set_trajectory_record(True)
            while not (done or cur_step == self.max_step):
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated = self.env.step_with_cmd_filter(action[0], action[1])

                done = terminated or truncated

                # Update the agent's current context
                self.sampler.update_context(obs=obs, action=action, reward=reward)

                cur_step += 1
                obs = next_obs
            
            # end to record trajectory
            self.env.set_trajectory_record(False)
            # record trajectory
            trajectories_list.append(self.env.get_trajectory())
            # save trajectory
            np.save(self.result_path + '/trajectories_list.npy', np.array(trajectories_list))

            

            # sample posterior z ~ q(z|c)
            self.agent.encoder.sample_z()
            
            # sample z ~ q(z|c) posterior
            self.agent.encoder.infer_posterior(self.agent.encoder.context)



        # for i in range(traj_num):
        #     trajs, num_samples = self.sampler.obtain_samples(
        #         task_index=task_index,
        #         accum_context=True,
        #     )

        #     # 根据 posterior 后验分布来更新 context
        #     # sample z ~ q(z|c) posterior
        #     self.agent.encoder.infer_posterior(self.agent.encoder.context)

        
    def visualize_within_tensorboard(self, test_results: Dict[str, Any], iteration: int) -> None:
        # 在 TensorBoard 上记录元训练和元测试结果
        # add meta test result
        self.writer.add_scalar("test/return_before_infer", test_results["return_before_infer"], iteration)
        self.writer.add_scalar("test/return_after_infer", test_results["return_after_infer"], iteration)

        self.return_before_infer.append(test_results["return_before_infer"])
        self.return_after_infer.append(test_results["return_after_infer"])
        np.save(self.result_path + '/return_before_infer.npy', np.array(self.return_before_infer))
        np.save(self.result_path + '/return_after_infer.npy', np.array(self.return_after_infer))

        # add meta train result
        self.writer.add_scalar("train/policy_loss", test_results["policy_loss"], iteration)
        self.writer.add_scalar("train/qf1_loss", test_results["qf1_loss"], iteration)
        self.writer.add_scalar("train/qf2_loss", test_results["qf2_loss"], iteration)
        self.writer.add_scalar("train/encoder_loss", test_results["encoder_loss"], iteration)
        self.writer.add_scalar("train/alpha_loss", test_results["alpha_loss"], iteration)
        self.writer.add_scalar("train/alpha", test_results["alpha"], iteration)
        self.writer.add_scalar("train/z_mean", test_results["z_mean"], iteration)
        self.writer.add_scalar("train/z_var", test_results["z_var"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

