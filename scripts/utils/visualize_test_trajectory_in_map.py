# import matplotlib
# matplotlib.use('pdf')  # 使用 PDF 后端
import matplotlib.pyplot as plt
import numpy as np

# parameter settings
num_raw = 2
num_col = 3


tree_map = np.load("/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/logs/test/2024-01-10-20-17-45/tree_map_list.npy", allow_pickle=True)
trajectories_list = np.load("/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/logs/test/2024-01-10-20-17-45/trajectories_list.npy", allow_pickle=True) 


# 绘制在同一张图中
# print(trajectories_list.shape)
# print(trajectories_list[0][0].shape)
# fig, axs = plt.subplots(num_raw, num_col, figsize=(16,4))
# print(axs.shape)

# for j in range(num_col):
#     for i in range(num_raw):
#         ax = axs[i, j]
#         # set title
#         if i == 0:
#             ax.set_title("Task %d Before Infer" %(j+1), fontsize=14)
#         elif i == 1:
#             ax.set_title("Task %d After Infer" %(j+1), fontsize=14)
#         tree_x = tree_map[:,0]
#         tree_y = tree_map[:,1]
#         tree_radius = tree_map[:,2]
#         dpi = 100
#         inches_per_meter = 0.54
#         radii_in_pixels = [radius / inches_per_meter* dpi for radius in tree_radius]
#         # plot tree
#         ax.scatter(tree_x, tree_y, s=radii_in_pixels, color='black', marker='o')

#         trj_i = j * num_raw + i
#         trajectories = trajectories_list[trj_i]
#         avoider_trj = trajectories[0]
#         pursuer_trj = trajectories[1]
#         if trj_i == 0:
#             ax.plot(avoider_trj[:,0], avoider_trj[:,1], label='avoider trajectory', color='red', linestyle='-')
#             ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], label='pursuer trajectory', color='blue', linestyle='--')
#         ax.plot(avoider_trj[:,0], avoider_trj[:,1], color='red', linestyle='-')
#         ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], color='blue', linestyle='--')
        
# # set title
# fig.suptitle(use_algo_name + " Trajectories on Training")
# # set legend
# fig.legend(loc='upper right')
# # for show
# plt.show()


# 分别绘制
def draw_filled_circle(center, radius, ax):
    circle = plt.Circle(center, radius, color='black', fill=True)
    ax.add_patch(circle)

print(trajectories_list.shape)
print(trajectories_list[0][0].shape)

tree_x = tree_map[:,0]
tree_y = tree_map[:,1]
tree_radius = tree_map[:,2]
center = [(x,y) for x,y in zip(tree_x, tree_y)]
print(center)

# figure 1
fig, ax = plt.subplots(figsize=(8, 6))
# plot tree
for i in range(len(center)):
    draw_filled_circle(center[i], tree_radius[i], ax)

# plot origin
pursuer_point = np.array([0,0])
ax.plot(pursuer_point[0], pursuer_point[1], 'b^', markersize=10, label="Pursuer Start Point")
avoider_point = np.array([7,0])
ax.plot(avoider_point[0], avoider_point[1], 'r*', markersize=10, label="Avoider Start Point")
# plot traj
traj_task1 = trajectories_list[0:2]
pre_infer_traj = traj_task1[0]
post_infer_traj = traj_task1[1]
avoider_trj = pre_infer_traj[0]
pursuer_trj = pre_infer_traj[1]
ax.plot(avoider_trj[:,0], avoider_trj[:,1], label='Avoider Pre-infer', color='red', linestyle='--')
ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], label='Pursuer Pre-infer', color='blue', linestyle='--')
avoider_trj = post_infer_traj[0]
pursuer_trj = post_infer_traj[1]
ax.plot(avoider_trj[:,0], avoider_trj[:,1], label='Avoider Post-infer', color='red', linestyle='-')
ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], label='Pursuer Post-infer', color='blue', linestyle='-')
ax.grid(True, linestyle='--')  # 设置网格为虚线
ax.set_xlabel("X", fontsize=14)  # 修改这里的 fig 为 ax
ax.set_ylabel("Y", fontsize=14)  # 修改这里的 fig 为 ax
ax.legend(loc="lower right", fontsize=10)
ax.set_aspect('equal')  #设置x,y轴相等
ax.set_xlim(-5.5, 20.5)
ax.set_ylim(-10.125, 10.125)
plt.savefig('/home/sjh/视频/task1_result.eps', dpi=400)
# plt.show()

# figure 2
fig, ax = plt.subplots(figsize=(8, 6))
# plot tree
for i in range(len(center)):
    draw_filled_circle(center[i], tree_radius[i], ax)

# plot origin
pursuer_point = np.array([0,0])
ax.plot(pursuer_point[0], pursuer_point[1], 'b^', markersize=10, label="Pursuer Start Point")
avoider_point = np.array([7,0])
ax.plot(avoider_point[0], avoider_point[1], 'r*', markersize=10, label="Avoider Start Point")
# plot traj
traj_task1 = trajectories_list[2:4]
pre_infer_traj = traj_task1[0]
post_infer_traj = traj_task1[1]
avoider_trj = pre_infer_traj[0]
pursuer_trj = pre_infer_traj[1]
ax.plot(avoider_trj[:,0], avoider_trj[:,1], label='Avoider Pre-infer', color='red', linestyle='--')
ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], label='Pursuer Pre-infer', color='blue', linestyle='--')
avoider_trj = post_infer_traj[0]
pursuer_trj = post_infer_traj[1]
ax.plot(avoider_trj[:,0], avoider_trj[:,1], label='Avoider Post-infer', color='red', linestyle='-')
ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], label='Pursuer Post-infer', color='blue', linestyle='-')
ax.grid(True, linestyle='--')  # 设置网格为虚线
ax.set_xlabel("X", fontsize=14)  # 修改这里的 fig 为 ax
ax.set_ylabel("Y", fontsize=14)  # 修改这里的 fig 为 ax
ax.legend(loc="lower right", fontsize=10)
ax.set_aspect('equal')  #设置x,y轴相等
ax.set_xlim(-5.5, 20.5)
ax.set_ylim(-10.125, 10.125)
plt.savefig('/home/sjh/视频/task2_result.eps', dpi=400)
# plt.show()

# figure 3
fig, ax = plt.subplots(figsize=(8, 6))
# plot tree
for i in range(len(center)):
    draw_filled_circle(center[i], tree_radius[i], ax)

# plot origin
pursuer_point = np.array([0,0])
ax.plot(pursuer_point[0], pursuer_point[1], 'b^', markersize=10, label="Pursuer Start Point")
avoider_point = np.array([7,0])
ax.plot(avoider_point[0], avoider_point[1], 'r*', markersize=10, label="Avoider Start Point")
# plot traj
traj_task1 = trajectories_list[4:6]
pre_infer_traj = traj_task1[1]
post_infer_traj = traj_task1[0]
avoider_trj = pre_infer_traj[0]
pursuer_trj = pre_infer_traj[1]
ax.plot(avoider_trj[:,0], avoider_trj[:,1], label='Avoider Pre-infer', color='red', linestyle='--')
ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], label='Pursuer Pre-infer', color='blue', linestyle='--')
avoider_trj = post_infer_traj[0]
pursuer_trj = post_infer_traj[1]
ax.plot(avoider_trj[:,0], avoider_trj[:,1], label='Avoider Post-infer', color='red', linestyle='-')
ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], label='Pursuer Post-infer', color='blue', linestyle='-')
ax.grid(True, linestyle='--')  # 设置网格为虚线
ax.set_xlabel("X", fontsize=14)  # 修改这里的 fig 为 ax
ax.set_ylabel("Y", fontsize=14)  # 修改这里的 fig 为 ax
ax.legend(loc="lower right", fontsize=10)
ax.set_aspect('equal')  #设置x,y轴相等
ax.set_xlim(-5.5, 20.5)
ax.set_ylim(-10.125, 10.125)
plt.savefig('/home/sjh/视频/task3_result.eps', dpi=400)
# plt.show()