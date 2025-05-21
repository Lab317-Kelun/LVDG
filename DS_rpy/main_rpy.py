import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from src.quat_class import quat_class
from src.util import load_tools, process_tools, quat_tools, plot_tools
import math


def rpy_to_quat(rpy_array):
    quat_array = []
    for rpy in rpy_array:
        r = R.from_euler('ZYX', rpy, degrees=False)
        quat = r.as_quat().tolist()
        quat_array.append(quat)
    return quat_array


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    roll, yaw = yaw, roll
    half_yaw = yaw / 2.0
    half_pitch = pitch / 2.0
    half_roll = roll / 2.0
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def downsample_to_fixed_size(data, target_size=500):
    original_size = len(data)
    if original_size <= target_size:
        return data
    indices = np.linspace(0, original_size - 1, target_size, dtype=int)
    downsampled_data = data[indices]
    return downsampled_data


def plot_rpy_trajectories(traj1, traj2, labels=["No Obstacle", "With Obstacle"]):
    rpy1 = R.from_quat(traj1).as_euler('ZYX', degrees=False)
    rpy2 = R.from_quat(traj2).as_euler('ZYX', degrees=False)
    t = np.arange(len(rpy1))
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    rpy_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs[i].plot(t, rpy1[:, i], label=labels[0], linestyle='--')
        axs[i].plot(t, rpy2[:, i], label=labels[1], linestyle='-')
        axs[i].set_ylabel(rpy_labels[i])
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel("Step")
    fig.suptitle("RPY Trajectory Comparison")
    plt.tight_layout()
    plt.show()


def visualize_3d_orientation_trajectories(rot_list_1, rot_list_2, interval=15):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Orientation Comparison (Z-axis Direction)", fontsize=14)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    def plot_trajectory_with_arrows(rot_list, color, label):
        trajectory = [np.zeros(3)]
        for i in range(1, len(rot_list)):
            direction = rot_list[i].apply([0, 0, 1])
            next_point = trajectory[-1] + direction * 0.05
            trajectory.append(next_point)

        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, label=label, alpha=0.6)

        for i in range(0, len(trajectory), interval):
            R_mat = rot_list[i].as_matrix()
            z_dir = R_mat[:, 2]
            ax.quiver(*trajectory[i], *z_dir, length=0.08, color=color, alpha=0.8)

    plot_trajectory_with_arrows(rot_list_1, 'blue', 'No Obstacle')
    plot_trajectory_with_arrows(rot_list_2, 'red', 'With Obstacle')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()


def main(start, end, task=''):
    coordinates = np.load(f'./data/combined_{task}_trajectories.npy', allow_pickle=True)
    rpy_data = np.load(f'./data/combined_{task}_quat.npy', allow_pickle=True)
    time_stamps = np.load('./data/combined_time_data.npy', allow_pickle=True)

    coordinates = [np.array(sublist) for sublist in coordinates]
    qua = [rpy_to_quat(rpy_array) for rpy_array in rpy_data]

    vel_array, quat = process_tools.compute_output(coordinates, qua, time_stamps)
    pose_init, quan_list, pose_final, quan_final = process_tools.extract_state(coordinates, qua)
    coordinates, qua, vel_array, quat = process_tools.rollout_list(coordinates, qua, vel_array, quat)
    quan_list = [R.from_quat(q) if isinstance(q, (list, np.ndarray)) else q for q in quan_list]

    quat_obj = quat_class(qua, quat, quan_final, dt=0.1, K_init=5)
    quat_obj.begin()

    rotation_list = quan_list[0]

    # === 无避障模拟 ===
    quat_obj.obstacle_rpy = None
    rpy_test, gamma_test, omega_test = quat_obj.sim(rotation_list, step_size=0.0012)
    quat_array_no_obs = [q.as_quat() for q in rpy_test]

    # === 有避障模拟 ===
    obstacle_rpy = [0, 0.209,-0.084]  # 可根据需求修改
    quat_obj.obstacle_rpy = obstacle_rpy
    new_q_test, new_gamma_test, new_omega_test = quat_obj.sim(rotation_list, step_size=0.0012)
    quat_array_with_obs = [q.as_quat() for q in new_q_test]

    # === RPY对比图 ===
    plot_rpy_trajectories(quat_array_no_obs, quat_array_with_obs, labels=["No Obstacle", "With Obstacle"])

    # === 保存避障后的结果 ===
    new_rpy_rad_array = [q.as_euler('ZYX', degrees=False) for q in new_q_test]
    new_rpy_rad_array = np.array(new_rpy_rad_array)
    downsampled_new_rpy_rad_array = downsample_to_fixed_size(new_rpy_rad_array, target_size=400)
    np.save(f'./data/{task}_rpy_results.npy', downsampled_new_rpy_rad_array)

    # === GMM轨迹图（如果已训练）===
    plot_tools.plot_gmm(coordinates, quat_obj.gmm)
    plt.show()

    # === 3D姿态对比图 ===
    visualize_3d_orientation_trajectories(rpy_test, new_q_test)


if __name__ == '__main__':
    start_rpy = R.from_quat(euler_to_quaternion(3.142, 0.209, -0.084))
    end_rpy = R.from_quat(euler_to_quaternion(-3.028, 0.196, 0.266))
    main(start_rpy, end_rpy, 'circle_close')
