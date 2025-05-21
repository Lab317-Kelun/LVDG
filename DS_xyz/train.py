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


def main():
    # 读取数据
    coordinates = np.load('./data/combined_trajectories.npy',allow_pickle=True)
    velocities = np.load('./data/combined_velocities.npy',allow_pickle=True)
    time_stamps = np.load('./data/combined_time_data.npy',allow_pickle=True)

    # 打印时间戳数据
    #print(time_stamps)

    # 定义一个函数，用于构建演示集
    manually_design_set_neum = construct_demonstration_set(coordinates, velocities, time_stamps, start=1, end=-1, gap=5)

    # ---------- 学习（加载）神经能量函数 NEUM ------------
    d_H = 10
    neum_learner = LearnNeum(manually_design_set=manually_design_set_neum, d_H=d_H, L_2=1e-6)
    print('--- 开始能量函数训练（加载）---')
    beta = 1.0
    save_path = 'NeumParameters/Neum_parameter_for_generated_data_beta' + str(beta) + '_dH' + str(d_H) + '.txt'
    # 训练或加载
    neum_parameters = neum_learner.train(save_path=save_path, beta=beta, maxiter=1500)
    neum_parameters = np.loadtxt(save_path)
    # print('--- 训练（加载）完成 ---')
    # print('绘制能量函数学习结果 ...')
    # neum_learner.show_learning_result(neum_parameters, num_levels=10)
    # print('绘制完成')

    # ------------------- 学习（加载）原始 ADS --------------------
    observation_noise = None
    gamma_oads = 0.3
    manually_design_set_oads = construct_demonstration_set(coordinates, velocities, time_stamps, start=1, end=-1, gap=5)
    ods_learner = LearnOds(manually_design_set=manually_design_set_oads, observation_noise=observation_noise,
                           gamma=gamma_oads)
    print('--- 开始原始 ADS 训练（加载） ---')
    save_path = 'OadsParameters/Oads_parameter_for_generated_data.txt'
    # 训练或加载，使用 "ods_learner.set_param" 加载参数
    ods_learner.train(save_path)
    oads_parameters = np.loadtxt(save_path)
    ods_learner.set_param(oads_parameters)
    print('--- 训练（加载）完成 ---')

    # ------------------- 形成稳定的 ADS ----------------------
    print('形成稳定的 ADS ...')
    sds_learner = LearnSds(lf_learner=neum_learner, ods_learner=ods_learner)

    # 定义函数 rho(x)，参见论文中的方程 (57)
    def func_rho(x):
        gamma = np.max(np.sqrt(np.sum(neum_learner.dot_x_set ** 2, axis=1))) / 1e3
        dvdx = neum_learner.dvdx(neum_parameters, x)
        return np.sqrt(np.dot(dvdx, dvdx)) * gamma

    # ---------- 运行稳定的 ADS ------------
    # 位置约束设置
    P = np.array([[1.96222637e-01, 0, 0], [0, 2.13162821e-14, 0], [0, 0, 3.60360360e-02]])  # 这里可以根据需要调整P矩阵
    r_thres = 0.0  # r_thres = 0.0 表示不使用位置约束
    eta = 0.05
    print('绘制稳定 ADS 结果 ...')
    sds_learner.plot_energy_field(lf_parameter=neum_parameters, func_rho=func_rho, P=P, r_thres=r_thres, eta=eta)
    sds_learner.plot_trajectory(lf_parameter=neum_parameters, func_rho=func_rho, P=P, r_thres=r_thres, eta=eta)
    print('绘制完成')


if __name__ == "__main__":
    main()
