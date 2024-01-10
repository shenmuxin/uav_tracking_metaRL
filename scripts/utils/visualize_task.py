import matplotlib.pyplot as plt
import numpy as np

# load data
task_trajs = np.load("/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/utils/results/trajectories_list.npy", allow_pickle=True)


plt.figure(figsize=(8, 6))
for i in range(len(task_trajs)):
    trajectories = task_trajs[i]
    avoider_trajs = trajectories[0]
    plt.plot(avoider_trajs[:, 0], avoider_trajs[:, 1], label="Task {}".format(i+1))
# plt.title("Avoider Trajectory of Different Task")
start_point = np.array([7,0])
plt.plot(start_point[0], start_point[1], 'r*', markersize=10, label="Start Point")

# # 设置方框的四个角点坐标
# x = [-20, 20, 20, -20, -20]
# y = [-20, -20, 20, 20, -20]
# # 绘制方框
# plt.plot(x, y, label='Maze', color='y')

plt.ylabel("Y", fontsize=14)
plt.xlabel("X", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--')  # 设置网格为虚线
plt.savefig('/home/sjh/视频/avoider_trajecotry_of_different_task.eps', dpi=400)
# plt.show()




# for i in range(len(task_trajs)):
#     trajectories = task_trajs[i]
#     avoider_trajs = trajectories[0]
#     plt.plot(avoider_trajs[:, 1], avoider_trajs[:, 0], label="Task {}".format(i+1))  # Swap x and y

# start_point = np.array([7,0])
# plt.plot(start_point[1], start_point[0], 'r*', markersize=10, label="Start Point")
# plt.title("Avoider Trajectories of Different Task")
# plt.xlabel("Y")  # Swap x and y labels
# plt.ylabel("X")
# plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio
# plt.legend()
# plt.show()
