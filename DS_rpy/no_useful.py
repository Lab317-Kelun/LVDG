import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from src.quat_class import quat_class
from src.util import plot_tools, load_tools, process_tools, quat_tools

# 加载数据
#p_raw, q_raw, t_raw, dt = load_tools.load_npy()
#print(dt)
p_raw, q_raw, t_raw = load_tools.load_clfd_dataset(task_id=0, num_traj=9, sub_sample=1)
# print('p_raw',p_raw)
# print('q_raw',q_raw)
# print('t_raw',t_raw)
#print(p_raw)
# p_raw, q_raw, t_raw = load_tools.load_demo_dataset()

# 数据预处理
p_in, q_in, t_in = process_tools.pre_process(p_raw, q_raw, t_raw, opt="savgol")
#print(q_in)
p_out, q_out = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out = process_tools.rollout_list(p_in, q_in, p_out, q_out)

# 初始化四元数类并开始计算
quat_obj = quat_class(q_in, q_out, q_att, dt=0.02, K_init=4)
quat_obj.begin()

# 四元数初始值
q_init = R.from_quat(-q_init[0].as_quat())
q_test, gamma_test, omega_test = quat_obj.sim(q_init, step_size=0.01)


# 打印输出轨迹的各个坐标点和四元数
print("输出轨迹的坐标点和四元数:")
for idx, (p, q) in enumerate(zip(p_out, q_out)):
    print(f"轨迹点 {idx + 1}:")
    print("坐标点:", p)
    if isinstance(q, list):
        print("四元数:", [quat.as_quat() for quat in q])
    else:
        print("四元数:", q.as_quat())
    print()

# 绘制结果
# 高斯混合模型（GMM）的聚类结果在三维空间中进行可视化
plot_tools.plot_gmm(p_in, quat_obj.gmm)
# plot_tools.plot_gamma(gamma_test)
# plot_tools.plot_omega(omega_test)

plt.show()