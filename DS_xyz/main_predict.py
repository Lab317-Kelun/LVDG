import numpy as np
import matplotlib.pyplot as plt
from Algorithms.Learn_NEUM import LearnNeum
from Algorithms.Learn_GPR_ODS import LearnOds
from Algorithms.Learn_SDS import LearnSds

np.random.seed(5)


def construct_demonstration_set(coordinates, velocities, time_stamps, start=1, end=-1, gap=5):
    n_tra = coordinates.shape[0]
    x_set = []
    dot_x_set = []
    t_set = []
    for i in range(n_tra):
        x_set.append(coordinates[i, start:end:gap])
        dot_x_set.append(velocities[i, start:end:gap])
        t_set.append(time_stamps[i, start:end:gap])

    x_set = np.array(x_set)
    dot_x_set = np.array(dot_x_set)
    t_set = np.array(t_set)
    return x_set, dot_x_set, t_set


def func_rho(x, neum_learner, neum_parameters):
    gamma = np.max(np.sqrt(np.sum(neum_learner.dot_x_set ** 2, axis=1))) / 1e3
    dvdx = neum_learner.dvdx(neum_parameters, x)
    return np.sqrt(np.dot(dvdx, dvdx)) * gamma


def generate_trajectory(start_point, goal_point, lf_parameter, sds_learner, func_rho, P=None, r_thres=0.0, eta=0.1):
    x = start_point
    period = 1e-2
    steps = int(100 / period)
    trajectory = [x]

    for step in range(steps):
        desired_v, _ = sds_learner.predict(lf_parameter, x, func_rho, P=P, r_thres=r_thres, eta=eta)
        x = x + desired_v * period
        trajectory.append(x)

        # 调试输出
        # print(f"Step: {step}, Position: {x}, Velocity: {desired_v}")

        if np.linalg.norm(x - goal_point) < 1e-2:
            print(f"Reached goal at step {step}")
            break

    return np.array(trajectory)


def main():
    # 读取数据
    coordinates = np.load('./data/combined_trajectories.npy', allow_pickle=True)
    velocities = np.load('./data/combined_velocities.npy', allow_pickle=True)
    time_stamps = np.load('./data/combined_time_data.npy', allow_pickle=True)

    # 定义一个函数，用于构建演示集
    manually_design_set_neum = construct_demonstration_set(coordinates, velocities, time_stamps, start=1, end=-1, gap=5)

    # ---------- 学习（加载）神经能量函数 NEUM ------------
    d_H = 10
    neum_learner = LearnNeum(manually_design_set=manually_design_set_neum, d_H=d_H, L_2=1e-6)
    print('--- 开始能量函数训练（加载）---')
    beta = 1.0
    save_path = 'NeumParameters/Neum_parameter_for_generated_data_beta' + str(beta) + '_dH' + str(d_H) + '.txt'
    # 训练或加载
    # neum_parameters = neum_learner.train(save_path=save_path, beta=beta, maxiter=1000)
    neum_parameters = np.loadtxt(save_path)
    print('--- 训练（加载）完成 ---')
    print('绘制能量函数学习结果 ...')
    #neum_learner.show_learning_result(neum_parameters, num_levels=10)
    print('绘制完成')

    # ------------------- 学习（加载）原始 ADS --------------------
    observation_noise = None
    gamma_oads = 0.5
    manually_design_set_oads = construct_demonstration_set(coordinates, velocities, time_stamps, start=1, end=-1, gap=5)
    ods_learner = LearnOds(manually_design_set=manually_design_set_oads, observation_noise=observation_noise,
                           gamma=gamma_oads)
    print('--- 开始原始 ADS 训练（加载） ---')
    save_path = 'OadsParameters/Oads_parameter_for_generated_data.txt'
    # 训练或加载，使用 "ods_learner.set_param" 加载参数
    # ods_learner.train(save_path)
    oads_parameters = np.loadtxt(save_path)
    ods_learner.set_param(oads_parameters)
    print('--- 训练（加载）完成 ---')

    # ------------------- 形成稳定的 ADS ----------------------
    print('形成稳定的 ADS ...')
    sds_learner = LearnSds(lf_learner=neum_learner, ods_learner=ods_learner)

    # 设置初末坐标点
    start_point = np.array([-0.249896,-0.011564,0.347132])
    goal_point = np.array([-0.098552,0.227625,0.351995])

    # 生成轨迹
    trajectory = generate_trajectory(start_point, goal_point, neum_parameters, sds_learner,
                                     lambda x: func_rho(x, neum_learner, neum_parameters))

    # 打印轨迹坐标点
    print("Trajectory_output:",trajectory)
    original_length = trajectory.shape[0]

    # 目标数据点数量
    target_length = 793

    # 计算步长
    step = original_length // target_length
    # 按照步长抽取样本
    downsampled_Trajectory = trajectory[::step]
    print(downsampled_Trajectory.shape)

    output_trajectory_path = './data/generated_trajectory.npy'
    np.save(output_trajectory_path, downsampled_Trajectory)
    print(f"Generated trajectory saved to {output_trajectory_path}")

    # 位置约束设置
    P = np.array([[1.96222637e-01, 0, 0], [0, 2.13162821e-14, 0], [0, 0, 3.60360360e-02]])  # 这里可以根据需要调整P矩阵
    r_thres = 0.0  # r_thres = 0.0 表示不使用位置约束
    eta = 0.05

    # 绘制能量场
    # print('绘制能量场 ...')
    # sds_learner.plot_energy_field(lf_parameter=neum_parameters,
    #                               func_rho=lambda x: func_rho(x, neum_learner, neum_parameters), P=P, r_thres=r_thres,
    #                               eta=eta)
    # plt.title('Energy Field')
    # plt.show()
    # print('绘制完成')

    # 绘制生成的轨迹和已知的轨迹
    print('绘制轨迹 ...')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', linewidth=2, alpha=1.0,
             label='Generated Trajectory')
    ax2.scatter(goal_point[0], goal_point[1], goal_point[2], c='green', alpha=1.0, s=50, marker='X', label='Goal Point')

    # 绘制已有的轨迹和点
    mark_size = 20
    ax2.scatter(0, 0, 0, c='black', alpha=1.0, s=mark_size, marker='X')
    ax2.scatter(sds_learner.lf_learner.x_set.reshape(-1, sds_learner.d_x)[:, 0],
                sds_learner.lf_learner.x_set.reshape(-1, sds_learner.d_x)[:, 1],
                sds_learner.lf_learner.x_set.reshape(-1, sds_learner.d_x)[:, 2],
                c='red', alpha=1.0, s=mark_size, marker='o')

    # 用黑色线连接红色点
    for traj in sds_learner.lf_learner.x_set:
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='black', linewidth=2, alpha=1.0)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.set_title('Generated and Known Trajectories')
    plt.show()
    print('绘制完成')


if __name__ == "__main__":
    main()
