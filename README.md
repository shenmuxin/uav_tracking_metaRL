# Quadrotor UAV Target Tracking and Obstacle Avoidance: meta-RL Method

## 1. Main Results

<video width="320" height="240" controls>
  <source src="figs/task1_whole_traj.mp4" type="video/mp4">
</video>


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
You can find many materials to teach you to install ROS Noetic. You can follow the official tutorial [Install ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu). Here just provide a tony version.

Add source
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

Set apt-key

```bash
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
```

Begin install

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full
```

Add path to `~/.bashrc`
```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Copy the robotic model in `/models` to `/PX4-Autopilot/Tools/sitl_gazebo/models`.
Copy the launch file in `/launch` to `/PX4-Autopilot/launch`.
Then follow the answer to add your custom robot as [here](https://discuss.px4.io/t/create-custom-model-for-sitl/6700).

### 2.3 Gazebo11
Download the [model](https://gazebosim.org/docs/all/getstarted), and then put it in `~/.gazebo`

### 2.3 Install PX4 and Mavros
This installation is a little complex. You can find the solution in official tutorial [Install PX4 and Mavros](https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html) or in [my blog](https://blog.csdn.net/qq_44940689/article/details/132827299#t8). Here just provide a tony version.

**Install PX4**

Clone the official repository

```bash
git clone https://github.com/PX4/PX4-Autopilot.git
```

Checkout to release version

```bash
cd PX4-Autopilot
git checkout origin/release/1.12
git submodule update --init --recursive
git status   
```

Add path to `~/.bashrc`

```bash
source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo
```

**Install Mavros**

```bash
sudo apt-get install ros-noetic-mavros ros-noetic-mavros-extras ros-noetic-mavros-msgs
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo chmod a+x ./install_geographiclib_datasets.sh
sudo ./install_geographiclib_datasets.sh
```

### 2.4 Install Anaconda

You can find the solution in official tutorial [Install Anaconda](https://www.anaconda.com/).


### 2.5 Create Virtual Environment

The default name of conda enviroment is `metarl`. You can change the name of virtual environment, but you should also change the corresponding place in the file `pearl2buf`, mainly in `pearl_infer_node.py` and `pearl_trainer_node.py`.

First create your virtual env.

```bash
conda create -n your_env_name python=3.8
conda activate my_env
```

Upgrade pip

```bash
pip install --upgrade pip
```

Install pkgs

```bash
pip install requirements.txt
```

There maybe some pkgs you can't install using the above command. You can install them manually.


## 3. Usage

## 3.1 Create Workspace

create workspace

```bash
mkdir your_workspace_name
cd your_workspace_name
mkdir src
cd src
catkin_init_workspace
```

This repository is a function package of the ROS frame. So, you should clone this repository to the local workspace putting it into file `src`, and then compile the whole workspace. 

```bash
cd ..
catkin_make
```

A quick Start:

- Change the hyperparameters in `scripts/pearl2buf/configs`
- Meta-training run script `scripts/pearl2buf/pearl_trainer_node.py`
- Meta-test run script `scripts/pearl2buf/pearl_infer_node.py`