import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from src.quat_class import quat_class
from src.util import load_tools, process_tools, quat_tools, plot_tools
import math

coordinates = np.load('./data/combined_trajectories.npy', allow_pickle=True)
rpy_data = np.load('./data/combined_quat.npy', allow_pickle=True)
velocities = np.load('./data/combined_velocities.npy', allow_pickle=True)
time_stamps = np.load('./data/combined_time_data.npy', allow_pickle=True)

# python 数组转 numpy数组
coordinates = [np.array(sublist) for sublist in coordinates]

# 将弧度值转换为角度值的函数
# def radians_to_degrees(nested_list):
#     return [[[math.degrees(value) for value in sublist] for sublist in lst] for lst in nested_list]
#
#
# # 将rpy_rad_array中的弧度值转换为角度值
# rpy_data = radians_to_degrees(rpy_data)


# 创建一个函数将 RPY 转换为四元数
def rpy_to_quat(rpy_array):
    quat_array = []
    for rpy in rpy_array:
        r = R.from_euler('ZYX', rpy, degrees=False)  # 从欧拉角创建旋转对象
        quat = r.as_quat().tolist()  # 获取四元数表示并转换为列表
        quat_array.append(quat)
    return quat_array


# rpy 转 四元数
qua = [rpy_to_quat(rpy_array) for rpy_array in rpy_data]
print("qua:",qua)
# 计算速度 输出的是坐标对应的速度
vel_array, quat = process_tools.compute_output(coordinates, qua, time_stamps)

# 从给定的多个轨迹中提取每条轨迹的初始状态（位置和四元数）以及第一条轨迹的最终状态（位置和四元数）
pose_init, quan_list, pose_final, quan_final = process_tools.extract_state(coordinates, qua)

# 将嵌套的列表展开成单个列表。具体来说，它将多个轨迹的数据（包括位置和四元数）合并成一个连续的列表
coordinates, qua, vel_array, quat = process_tools.rollout_list(coordinates, qua, vel_array, quat)

# 确保 quan_list 的每个元素都是 Rotation 对象
quan_list = [R.from_quat(q) if isinstance(q, (list, np.ndarray)) else q for q in quan_list]

# 初始化四元数类并开始计算
quat_obj = quat_class(qua, quat, quan_final, dt=0.2, K_init=4)
quat_obj.begin()

# 设置四元数初始值 四元数 转 rpy
rotation_list = quan_list[0]  # 这里不需要再调用 as_quat，因为 quan_list 已经是 Rotation 对象

# 进行仿真，rpy训练
rpy_test, gamma_test, omega_test = quat_obj.sim(rotation_list, step_size=0.0012)

# 打印仿真结果的 RPY 值
# for i, rot in enumerate(rpy_test):
#     rpy = rot.as_euler('xyz', degrees=True)
#     rpy_rad = np.deg2rad(rpy)  # 转换为弧度
#     print(f"Rotation {i}: RPY (Degrees) = {rpy}, RPY (Radians) = {rpy_rad}")

plot_tools.plot_gmm(coordinates, quat_obj.gmm)
plt.show()

# 新的始末坐标点和四元数
new_start_quat = R.from_quat([  0.018,   0.999,   0.023,  -0.022])
new_end_quat = R.from_quat([ 0.037, -0.734,  0.678, 0.010])

# 更新四元数类的输入和目标
quat_obj.elasticUpdate([new_start_quat], [new_end_quat], quat_obj.gmm, quat_obj.q_att)

# 进行新的仿真，获取新的轨迹
new_q_test, new_gamma_test, new_omega_test = quat_obj.sim(new_start_quat, step_size=0.0005)

# 创建用于存储新的弧度制 RPY 值的数组
new_rpy_rad_array = []

# 打印新的仿真结果的 RPY 值，并存储为弧度制
for i, rot in enumerate(new_q_test):
    rpy = rot.as_euler('ZYX', degrees=True)
    rpy_rad = np.deg2rad(rpy)  # 转换为弧度z
    new_rpy_rad_array.append(rpy_rad)
    print(f"Rotation {i}: RPY (Degrees) = {rpy}, RPY (Radians) = {rpy_rad}")

# 将新的弧度制的 RPY 值转换为 NumPy 数组并保存为 .npy 文件
new_rpy_rad_array = np.array(new_rpy_rad_array)
np.save('./rpy_results.npy', new_rpy_rad_array)

plot_tools.plot_gmm(coordinates, quat_obj.gmm)
plt.show()
