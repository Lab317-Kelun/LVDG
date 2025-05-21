import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .util import optimize_tools, quat_tools
from .gmm_class import gmm_class

def _write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def compute_ang_vel(q_k, q_kp1, dt=0.01):
    dq = q_kp1 * q_k.inv()
    dq = dq.as_rotvec()
    w = dq / dt
    return w

class quat_class:
    def __init__(self, q_in: list, q_out: list, q_att: R, dt, K_init: int) -> None:
        if isinstance(q_in[0], (list, np.ndarray)):
            q_in = [R.from_quat(q) for q in q_in]
        if isinstance(q_out[0], (list, np.ndarray)):
            q_out = [R.from_quat(q) for q in q_out]
        if isinstance(q_att, (list, np.ndarray)):
            q_att = R.from_quat(q_att)

        self.q_in = q_in
        self.q_out = q_out
        self.q_att = q_att
        self.dt = dt
        self.K_init = K_init
        self.M = len(q_in)
        self.N = 4

        self.tol = 1e-2
        self.max_iter = 5000

        self.obstacle_rpy = [0,0,0]

        file_path = os.path.dirname(os.path.realpath(__file__))
        self.output_path = os.path.join(os.path.dirname(file_path), 'output_ori.json')

    def _cluster(self):
        gmm = gmm_class(self.q_in, self.q_att, self.K_init)
        self.gamma = gmm.fit()
        self.K = gmm.K
        self.gmm = gmm

    def _optimize(self):
        A_ori = optimize_tools.optimize_ori(self.q_in, self.q_out, self.q_att, self.gamma)

        q_in_dual = [R.from_quat(-q.as_quat()) for q in self.q_in]
        q_out_dual = [R.from_quat(-q.as_quat()) for q in self.q_out]
        q_att_dual = R.from_quat(-self.q_att.as_quat())
        A_ori_dual = optimize_tools.optimize_ori(q_in_dual, q_out_dual, q_att_dual, self.gamma)

        self.A_ori = np.concatenate((A_ori, A_ori_dual), axis=0)

    def begin(self):
        self._cluster()
        self._optimize()

    def elasticUpdate(self, new_q_in, new_q_out, gmm_struct_ori, att_ori_new):
        if isinstance(new_q_in[0], (list, np.ndarray)):
            new_q_in = [R.from_quat(q) for q in new_q_in]
        if isinstance(new_q_out[0], (list, np.ndarray)):
            new_q_out = [R.from_quat(q) for q in new_q_out]
        if isinstance(att_ori_new, (list, np.ndarray)):
            att_ori_new = R.from_quat(att_ori_new)

        self.q_att = att_ori_new

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

    def sim(self, q_init, step_size):
        q_test = [q_init]
        gamma_test = []
        omega_test = []

        i = 0
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol:
            if i > self.max_iter:
                break

            q_in = q_test[i]
            q_next, gamma, omega = self._step(q_in, step_size)

            q_test.append(q_next)
            gamma_test.append(gamma[:, 0])
            omega_test.append(omega)
            i += 1

        return q_test, np.array(gamma_test), np.array(omega_test)

    def _step(self, q_in, step_size, avoid_scale=0.1):
        A_ori = self.A_ori
        q_att = self.q_att
        K = self.K
        gmm = self.gmm
        obstacle_rpy = self.obstacle_rpy

        gamma = gmm.logProb(q_in)

        q_out_att = np.zeros((4, 1))
        q_diff = quat_tools.riem_log(q_att, q_in)
        # print('q_diff',q_diff)
        for k in range(K):
            q_out_att += gamma[k, 0] * A_ori[k] @ q_diff.T
        q_out_body = quat_tools.parallel_transport(q_att, q_in, q_out_att.T)

        # 避障扰动加入主方向
        if obstacle_rpy is not None and len(obstacle_rpy) > 0:
            q_obs = R.from_euler('ZYX', obstacle_rpy, degrees=False).as_quat()
            q_obs_log = quat_tools.riem_log(q_in, R.from_quat(q_obs))
            q_out_body += avoid_scale * q_obs_log

        q_out_q = quat_tools.riem_exp(q_in, q_out_body)
        q_out = R.from_quat(q_out_q.reshape(4,))
        omega = compute_ang_vel(q_in, q_out, self.dt)

        # Dual部分也加扰动
        q_att_dual = R.from_quat(-q_att.as_quat())
        q_out_att_dual = np.zeros((4, 1))
        q_diff_dual = quat_tools.riem_log(q_att_dual, q_in)
        for k in range(K):
            q_out_att_dual += gamma[K + k, 0] * A_ori[K + k] @ q_diff_dual.T
        q_out_body_dual = quat_tools.parallel_transport(q_att_dual, q_in, q_out_att_dual.T)

        if obstacle_rpy is not None and len(obstacle_rpy) > 0:
            q_out_body_dual += avoid_scale * (q_obs_log)

        q_out_q_dual = quat_tools.riem_exp(q_in, q_out_body_dual)
        q_out_dual = R.from_quat(q_out_q_dual.reshape(4,))
        omega += compute_ang_vel(q_in, q_out_dual, self.dt)

        q_next = R.from_rotvec(omega * step_size) * q_in
        return q_next, gamma, omega

    def _logOut(self, write_json, *args):
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
