# Quadrotor UAV Target Tracking and Obstacle Avoidance: meta-RL Method

## 1. Main Results


## 2. Installation
### 2.1 Software Version

| Interface                       | Version                    |
| -------------------------- | ----------------------- |
| Ubuntu           | 20.04 |
| ROS              | noetic     |
| PX4               | v1.12     |
| pytorch          |  2.0.1     |
| cuda              | 11.7     |


### 2.2 Install ROS Noetic
You can find many materials to teach you to install ROS Noetic. You can follow the official tutorial [Install ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu).

### 2.3 Install PX4 and Mavros
This installation is a little complex. You can find the solution in official tutorial [Install PX4 and Mavros](https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html) or in [my blog](https://blog.csdn.net/qq_44940689/article/details/132827299#t8).


### 2.4 Install Pytorch
You can find the solution in official tutorial [Install Pytorch](https://pytorch.org/get-started/locally/).

## 3. Usage

This repository is a function package of the ROS frame. So, you should clone this repository to the local workspace, and then compile the whole workspace. The default name of conda enviroment is `metarl`. You can change the name of virtual environment, but you should also change the corresponding place in the file `pearl2buf`, mainly in `pearl_infer_node.py` and `pearl_trainer_node.py`.

A quick Start:

- Change the hyperparameters in `scripts/pearl2buf/configs`
- Meta-training run script `scripts/pearl2buf/pearl_trainer_node.py`
- Meta-test run script `scripts/pearl2buf/pearl_infer_node.py`