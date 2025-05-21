import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from src.quat_class import quat_class
from src.util import load_tools, process_tools, quat_tools, plot_tools
import math

# 创建一个函数将 RPY 转换为四元数
def rpy_to_quat(rpy_array):
    quat_array = []
    for rpy in rpy_array:
        r = R.from_euler('ZYX', rpy, degrees=False)  # 从欧拉角创建旋转对象
        quat = r.as_quat().tolist()  # 获取四元数表示并转换为列表
        quat_array.append(quat)
    return quat_array


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """
    将弧度制欧拉角 (roll, pitch, yaw) 按照 ZYX 顺序转换为四元数。

    参数：
        roll: 绕 X 轴的旋转角度（弧度）
        pitch: 绕 Y 轴的旋转角度（弧度）
        yaw: 绕 Z 轴的旋转角度（弧度）

    返回：
        四元数 (w, x, y, z)
    """
    # 按照zyx的顺序旋转需要把roll和yaw调换
    roll, yaw = yaw, roll

    # 计算每个角的一半
    half_yaw = yaw / 2.0
    half_pitch = pitch / 2.0
    half_roll = roll / 2.0

    # 计算正弦和余弦
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)

    # 计算四元数分量
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w


def downsample_to_fixed_size(data, target_size=500):
    """
    将任意长度的数据均匀下采样到指定数量的点。

    参数:
    - data: 原始数据（1D NumPy 数组或列表）。
    - target_size: 目标数据点数量（默认为 500）。

    返回:
    - 下采样后的数据（1D NumPy 数组）。

    # 示例数据（假设原始数据长度为 874 个点）
    original_data = np.linspace(800, 1000, 874)  # 生成 874 个点的数组

    # 均匀下采样到 500 个点
    downsampled_data = downsample_to_fixed_size(original_data, 500)

    # 输出结果
    print("原始数据长度:", len(original_data))
    print("下采样数据长度:", len(downsampled_data))
    """
    original_size = len(data)

    # 如果原始数据小于或等于目标大小，直接返回原始数据
    if original_size <= target_size:
        return data

    # 计算均匀采样的索引
    indices = np.linspace(0, original_size - 1, target_size, dtype=int)

    # 根据索引提取采样后的数据
    downsampled_data = data[indices]
    return downsampled_data


def main(start, end, task=''):
    start = R.from_quat(euler_to_quaternion(*start))
    end = R.from_quat(euler_to_quaternion(*end))

    coordinates = np.load(f'./data/combined_{task}_trajectories.npy', allow_pickle=True)
    rpy_data = np.load(f'./data/combined_{task}_quat.npy', allow_pickle=True)
    # velocities = np.load(f'./data/combined_{task}_velocities.npy', allow_pickle=True)
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
    
    # rpy 转 四元数
    qua = [rpy_to_quat(rpy_array) for rpy_array in rpy_data]
    # print(qua)
    # 计算速度 输出的是坐标对应的速度
    vel_array, quat = process_tools.compute_output(coordinates, qua, time_stamps)
    
    # 从给定的多个轨迹中提取每条轨迹的初始状态（位置和四元数）以及第一条轨迹的最终状态（位置和四元数）
    pose_init, quan_list, pose_final, quan_final = process_tools.extract_state(coordinates, qua)
    
    # 将嵌套的列表展开成单个列表。具体来说，它将多个轨迹的数据（包括位置和四元数）合并成一个连续的列表
    coordinates, qua, vel_array, quat = process_tools.rollout_list(coordinates, qua, vel_array, quat)
    
    # 确保 quan_list 的每个元素都是 Rotation 对象
    quan_list = [R.from_quat(q) if isinstance(q, (list, np.ndarray)) else q for q in quan_list]
    
    # 初始化四元数类并开始计算
    quat_obj = quat_class(qua, quat, quan_final, dt=0.1, K_init=4)
    quat_obj.begin()
    
    # 设置四元数初始值 四元数 转 rpy
    rotation_list = quan_list[0]  # 这里不需要再调用 as_quat，因为 quan_list 已经是 Rotation 对象
    
    # 进行仿真，rpy训练
    # rpy_test, gamma_test, omega_test = quat_obj.sim(rotation_list, step_size=0.0012)
    
    # 打印仿真结果的 RPY 值
    # for i, rot in enumerate(rpy_test):
    #     rpy = rot.as_euler('xyz', degrees=True)
    #     rpy_rad = np.deg2rad(rpy)  # 转换为弧度
    #     print(f"Rotation {i}: RPY (Degrees) = {rpy}, RPY (Radians) = {rpy_rad}")

    #TODO
    # plot_tools.plot_gmm(coordinates, quat_obj.gmm)
    # plt.show()
    
    # 新的始末坐标点和四元数
    new_start_quat = start
    new_end_quat = end
    
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
        # print(f"Rotation {i}: RPY (Degrees) = {rpy}, RPY (Radians) = {rpy_rad}")

    # 将新的弧度制的 RPY 值转换为 NumPy 数组并保存为 .npy 文件
    new_rpy_rad_array = np.array(new_rpy_rad_array)

    # 目标数据点数量
    target_length = 400

    downsampled_new_rpy_rad_array = downsample_to_fixed_size(new_rpy_rad_array, target_length)
    #TODO
    # print(downsampled_new_rpy_rad_array.shape)
    # np.set_printoptions(threshold=np.inf)
    # print(downsampled_new_rpy_rad_array)
    np.save(f'./RMDemo_Moves/src/core/{task}_rpy_results.npy', downsampled_new_rpy_rad_array)

    #TODO
    # plot_tools.plot_gmm(coordinates, quat_obj.gmm)
    # plt.show()
    
    
if __name__ == '__main__':
    start_rpy = R.from_quat(euler_to_quaternion(2.892,0.51,-0.271))
    end_rpy = R.from_quat(euler_to_quaternion(2.64,-1.41,-2.237))
    main(start_rpy, end_rpy, 'put')

    # 示例
    # roll = np.radians(30)  # 绕 X 轴旋转 30 度
    # pitch = np.radians(45)  # 绕 Y 轴旋转 45 度
    # yaw = np.radians(60)  # 绕 Z 轴旋转 60 度
    #
    # quaternion = euler_to_quaternion(roll, pitch, yaw)
    # print(euler_to_quaternion(3.09,0.312,-0.033), end_rpy)
    # print("四元数:", quaternion)
