import matplotlib.pyplot as plt
import numpy as np

# load data
pearl_before1 = np.load("/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/logs/train/2023-12-16-21-31-40/return_before_infer.npy")
pearl_after1 = np.load("/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/logs/train/2023-12-16-21-31-40/return_after_infer.npy")

pearl_before2 = np.load("/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/logs/train/2023-12-18-10-48-21/return_before_infer.npy")
pearl_after2 = np.load("/home/sjh/catkin_ws/src/meta_rl_tracking/scripts/logs/train/2023-12-18-10-48-21/return_after_infer.npy")


pearl_before = np.concatenate((pearl_before1, pearl_before2[:50]), axis=0)
pearl_after = np.concatenate((pearl_after1, pearl_after2[:50]), axis=0)


pearl_return = (pearl_after + pearl_before) / 2

iterations = np.arange(1, pearl_before.shape[0]+1)

def exponential_moving_average(data, alpha):
    """
    func->
        calculate exponential moving average
    """
    ema = []  
    ema.append(data[0])  # the first point
    for i in range(1, len(data)):
        ema_value = (1 - alpha) * data[i] +  alpha * ema[-1]
        ema.append(ema_value)
    return ema

# calculate ema
pearl_before_ema = exponential_moving_average(pearl_before, 0.85)
pearl_after_ema = exponential_moving_average(pearl_after, 0.85)
pearl_return_ema = exponential_moving_average(pearl_return, 0.85)

# calculate uncertain range
pearl_before_std_dev = np.std(pearl_before_ema)
pearl_after_std_dev = np.std(pearl_after_ema)
pearl_return_std_dev = np.std(pearl_return_ema)



# setting background color
fig, ax = plt.subplots(figsize=(8, 6))
# ax.set_facecolor('lightsteelblue')
# ax.patch.set_alpha(0.1)  # 设置透明度

# plot uncertain area
ax.fill_between(iterations, pearl_before_ema - pearl_before_std_dev, pearl_before_ema + pearl_before_std_dev, color='lightgreen', alpha=0.15, edgecolor='none')
ax.fill_between(iterations, pearl_after_ema - pearl_after_std_dev, pearl_after_ema + pearl_after_std_dev, color='moccasin', alpha=0.15, edgecolor='none')
ax.fill_between(iterations, pearl_return_ema - pearl_return_std_dev, pearl_return_ema + pearl_return_std_dev, color='lightsteelblue', alpha=0.15, edgecolor='none')



# plot return curve
ax.plot(iterations, pearl_before_ema, label='Pre-inference', color='darkgreen', linewidth=1.5)
ax.plot(iterations, pearl_after_ema, label='Post-inference', color='darkorange', linewidth=1.5)
ax.plot(iterations, pearl_return_ema, label='Average Return', color='blue', linewidth=1.5)

# add grid
ax.grid(True, linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Returns', fontsize=14)
ax.set_xlabel('Meta Iterations', fontsize=14)
ax.legend(loc="lower right", fontsize=10)
plt.savefig('/home/sjh/视频/meta_training_results.eps', dpi=400)
# plt.show()






