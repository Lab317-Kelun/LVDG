import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .util import optimize_tools, quat_tools
from .gmm_class import gmm_class

# 定义一个函数，用于将数据写入JSON文件
def _write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)

# 计算角速度的函数
def compute_ang_vel(q_k, q_kp1, dt=0.01):
    """ 计算角速度 """
    # 使用固定坐标系下的四元数差分
    dq = q_kp1 * q_k.inv()
    # 将四元数差分转换为旋转向量
    dq = dq.as_rotvec()
    # 计算角速度
    w = dq / dt
    return w

# 定义四元数类
class quat_class:
    def __init__(self, q_in: list, q_out: list, q_att: R, dt, K_init: int) -> None:
        """
        参数:
        ----------
            q_in (list):            表示输入方向的旋转对象列表
            q_out (list):           表示输出方向的旋转对象列表
            q_att (Rotation):       表示吸引器方向的单个旋转对象
            dt:                     表示时间差
            K_init:                 初始的高斯分量数量
        """
        # 确保输入的四元数是 `Rotation` 对象
        if isinstance(q_in[0], (list, np.ndarray)):
            q_in = [R.from_quat(q) for q in q_in]
        if isinstance(q_out[0], (list, np.ndarray)):
            q_out = [R.from_quat(q) for q in q_out]
        if isinstance(q_att, (list, np.ndarray)):
            q_att = R.from_quat(q_att)

        # 存储参数
        self.q_in = q_in
        self.q_out = q_out
        self.q_att = q_att
        self.dt = dt
        self.K_init = K_init
        self.M = len(q_in)
        self.N = 4

        # 仿真参数
        self.tol = 10E-3
        self.max_iter = 5000

        # 定义输出路径
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.output_path = os.path.join(os.path.dirname(file_path), 'output_ori.json')

    # 聚类函数
    def _cluster(self):
        gmm = gmm_class(self.q_in, self.q_att, self.K_init)  # 创建GMM模型实例
        self.gamma = gmm.fit()  # 使用GMM模型对数据进行拟合并获得分类结果
        self.K = gmm.K  # 更新类别数
        self.gmm = gmm  # 存储GMM模型实例

    # 优化函数
    def _optimize(self):
        # 对原始四元数数据进行优化
        A_ori = optimize_tools.optimize_ori(self.q_in, self.q_out, self.q_att, self.gamma)

        # 生成对偶四元数数据
        q_in_dual = [R.from_quat(-q.as_quat()) for q in self.q_in]
        q_out_dual = [R.from_quat(-q.as_quat()) for q in self.q_out]
        q_att_dual = R.from_quat(-self.q_att.as_quat())

        # 对对偶四元数数据进行优化
        A_ori_dual = optimize_tools.optimize_ori(q_in_dual, q_out_dual, q_att_dual, self.gamma)

        # 合并优化结果
        self.A_ori = np.concatenate((A_ori, A_ori_dual), axis=0)

    # 开始聚类和优化
    def begin(self):
        self._cluster()
        self._optimize()

    # 弹性更新函数
    def elasticUpdate(self, new_q_in, new_q_out, gmm_struct_ori, att_ori_new):
        if isinstance(new_q_in[0], (list, np.ndarray)):
            new_q_in = [R.from_quat(q) for q in new_q_in]
        if isinstance(new_q_out[0], (list, np.ndarray)):
            new_q_out = [R.from_quat(q) for q in new_q_out]
        if isinstance(att_ori_new, (list, np.ndarray)):
            att_ori_new = R.from_quat(att_ori_new)

        self.q_att = att_ori_new

        # 检查 gmm_struct_ori 的类型并选择合适的访问方式
        if isinstance(gmm_struct_ori, dict):
            Prior = gmm_struct_ori["Prior"]
            Mu = gmm_struct_ori["Mu"]
            Sigma = gmm_struct_ori["Sigma"]
        else:
            Prior = gmm_struct_ori.Prior
            Mu = gmm_struct_ori.Mu
            Sigma = gmm_struct_ori.Sigma

        gamma = self.gmm.elasticUpdate(new_q_in, Prior, Mu, Sigma)

        A_ori = optimize_tools.optimize_ori(new_q_in, new_q_out, self.q_att, gamma)

        q_in_dual = [R.from_quat(-q.as_quat()) for q in new_q_in]
        q_out_dual = [R.from_quat(-q.as_quat()) for q in new_q_out]
        q_att_dual = R.from_quat(-self.q_att.as_quat())
        A_ori_dual = optimize_tools.optimize_ori(q_in_dual, q_out_dual, q_att_dual, gamma)

        self.A_ori = np.concatenate((A_ori, A_ori_dual), axis=0)

    # 仿真函数
    def sim(self, q_init, step_size):
        '''
            函数中的变量 暂未改变 该函数功能为
            模拟一个rpy迭代的演化过程，直到系统达到某个终止条件或达到最大迭代次数。
            它通过不断更新当前rpy，计算下一步的rpy、gamma 和 omega，并将这些值存储在相应的列表中。
            最终返回所有的rpy状态、gamma 和 omega 的历史值。
        '''
        q_test = [q_init]
        gamma_test = []
        omega_test = []

        i = 0
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol:
            if i > self.max_iter:
                # print("超过最大迭代次数")
                break

            q_in = q_test[i]

            q_next, gamma, omega = self._step(q_in, step_size)

            q_test.append(q_next)
            gamma_test.append(gamma[:, 0])
            omega_test.append(omega)

            i += 1

        return q_test, np.array(gamma_test), np.array(omega_test)

    # 单步仿真函数
    def _step(self, q_in, step_size):
        """ 向前积分一步 """

        # 读取参数
        A_ori = self.A_ori  # (2K, N, N)
        q_att = self.q_att
        K = self.K
        gmm = self.gmm

        # 计算gamma
        gamma = gmm.logProb(q_in)  # (2K, 1)

        # 初次覆盖
        q_out_att = np.zeros((4, 1))
        q_diff = quat_tools.riem_log(q_att, q_in)
        for k in range(K):
            q_out_att += gamma[k, 0] * A_ori[k] @ q_diff.T
        q_out_body = quat_tools.parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q = quat_tools.riem_exp(q_in, q_out_body)
        q_out = R.from_quat(q_out_q.reshape(4, ))
        omega = compute_ang_vel(q_in, q_out, self.dt)

        # 双重覆盖
        q_att_dual = R.from_quat(-q_att.as_quat())
        q_out_att_dual = np.zeros((4, 1))
        q_diff_dual = quat_tools.riem_log(q_att_dual, q_in)
        for k in range(K):
            q_out_att_dual += gamma[self.K + k, 0] * A_ori[self.K + k] @ q_diff_dual.T
        q_out_body_dual = quat_tools.parallel_transport(q_att_dual, q_in, q_out_att_dual.T)
        q_out_q_dual = quat_tools.riem_exp(q_in, q_out_body_dual)
        q_out_dual = R.from_quat(q_out_q_dual.reshape(4, ))
        omega += compute_ang_vel(q_in, q_out_dual, self.dt)

        # 向前传播
        q_next = R.from_rotvec(omega * step_size) * q_in  # 在世界坐标系中合成

        return q_next, gamma, omega

    # 输出日志
    def _logOut(self, write_json, *args):
        # print(args)

        Prior = self.gmm.Prior
        Mu = self.gmm.Mu
        Mu_rollout = [q_mean.as_quat() for q_mean in Mu]
        Sigma = self.gmm.Sigma

        Mu_arr = np.zeros((2 * self.K, self.N))
        Sigma_arr = np.zeros((2 * self.K, self.N, self.N), dtype=np.float32)

        for k in range(2 * self.K):
            Mu_arr[k, :] = Mu_rollout[k]
            Sigma_arr[k, :, :] = Sigma[k]

        json_output = {
            "name": "Quaternion-DS",
            "K": self.K,
            "M": 4,
            "Prior": Prior,
            "Mu": Mu_arr.ravel().tolist(),
            "Sigma": Sigma_arr.ravel().tolist(),
            'A_ori': self.A_ori.ravel().tolist(),
            'att_ori': self.q_att.as_quat().ravel().tolist(),
            "dt": self.dt,
            'q_0': self.q_in[0].as_quat().ravel().tolist(),
            "gripper_open": 0
        }

        if write_json:
            if len(args) == 0:
                _write_json(json_output, self.output_path)
            else:
                _write_json(json_output, os.path.join(args[0], '1.json'))

        return json_output