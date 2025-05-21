import numpy as np

def main(task=''):
    # 加载数据
    trajectory_array = np.load(f'RMDemo_Moves/src/core/generated_{task}_trajectory.npy')
    # 获取数组的最后一个元素
    last_element = trajectory_array[-1]

    # 将最后一个元素添加到数组的末尾
    trajectory_array = np.vstack([trajectory_array, last_element])
    rpy_array = np.load(f'RMDemo_Moves/src/core/{task}_rpy_results.npy')
    # 检查数据形状
    trajectory_len = trajectory_array.shape[0]
    rpy_len = rpy_array.shape[0]
    # print(rpy_len)
    # 计算步长
    step = trajectory_len // rpy_len

    # 按照步长抽取样本
    selected_indices = np.arange(0, step * rpy_len, step)
    downsampled_trajectory_array = trajectory_array[selected_indices]

    # 确保抽取后的数组与 rpy_array 的条目数相同
    assert downsampled_trajectory_array.shape[0] == rpy_len, "Downsampled trajectory array length does not match RPY array length"

    # 在每组数据后面添加相应的 RPY 值
    trajectory_with_rpy = np.concatenate((downsampled_trajectory_array, rpy_array), axis=1)

    # 计算新的步长，以进一步下采样
    final_sample_count = 100 # 假设你需要最终减少到 100 个数据点 这里会有点小问题 实际减少不到所填的数据 但不用管它 机械臂能运行就好
    new_step = max(1, len(trajectory_with_rpy) // final_sample_count)

    # 按照新的步长抽取样本
    further_downsampled_trajectory = trajectory_with_rpy[::new_step]
    # 剔除前5个数据
    further_downsampled_trajectory = further_downsampled_trajectory[5:]

    #TODO
    # print(further_downsampled_trajectory.shape)
    # 打印下采样结果
    # print("Further downsampled trajectory with RPY:\n",  further_downsampled_trajectory)

    # 打印结果
    # print("Trajectory with RPY:\n", trajectory_with_rpy)

    # 如果需要保存结果，可以使用 np.save
    # np.save('trajectory_with_rpy.npy', trajectory_with_rpy)
    np.save(f'RMDemo_Moves/src/core/further_downsampled_{task}_trajectory_with_rpy.npy', further_downsampled_trajectory)


if __name__ == '__main__':
    main('close')
