import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from . import plot_tools, optimize_tools, quat_tools
from .quat_tools import *
from .plot_tools import *


def _compute_ang_vel(q_i, q_ip1, dt=0.01):
    """  Compute angular velocity """

    # dq = q_i.inv() * q_ip1    # from q_i to q_ip1 in body frame
    dq = q_ip1 * q_i.inv()  # from q_i to q_ip1 in fixed frame

    dq = dq.as_rotvec()
    w = dq / dt

    return w


def _shift_pos(p_list):
    L = len(p_list)

    p_att_list = [p_list[l][-1, :] for l in range(L)]
    p_att_mean = np.mean(np.array(p_att_list), axis=0)

    p_shifted = []
    for l in range(L):
        p_diff = p_att_mean - p_att_list[l]

        p_shifted.append(p_diff.reshape(1, -1) + p_list[l])

    return p_shifted, p_att_mean


def _shift_ori(q_list):
    """
    Note:
    ---- 
        Scipy methods, e.g. "R.mean()", "R.inv()" and "R.__mul__()" will OFTEN flip the SIGNS of the computed quaternion
        
        Do NOT output "q_att_mean" as the ATTRACTOR which could be SIGN-inconsistent with the rest of quaternions
        INSTEAD, always output the LAST of the shifted quaternions
    """

    L = len(q_list)

    q_att_list = [q_list[l][-1] for l in range(L)]
    q_att_mean = R.from_quat([q_att_list[l].as_quat() for l in range(L)]).mean()

    q_shifted = []
    for l in range(L):
        q_diff = q_att_mean * q_att_list[l].inv()

        q_shifted.append([q_diff * q for q in q_list[l]])

    return q_shifted, q_shifted[-1][-1]


def _smooth_ori(q_list, q_att, opt):
    """
    Smoothen the orientation trajectory using Savgol filter or SLERP interpolation

    Note:
    ----
        The value of k are parameters that can be tuned in both methods
    """

    L = len(q_list)

    q_smooth = []
    for l in range(L):
        q_l = q_list[l]

        if opt == "savgol":
            k = 80

            q_l_att = quat_tools.riem_log(q_att, q_l)

            q_smooth_att = savgol_filter(q_l_att, window_length=k, polyorder=2, axis=0, mode="nearest")

            q_smooth_arr = quat_tools.riem_exp(q_att, q_smooth_att)

            q_smooth.append([R.from_quat(q_smooth_arr[i, :]) for i in range(q_smooth_arr.shape[0])])


        elif opt == "slerp":
            k = 40

            t_list = [0.1 * i for i in range(len(q_l))]

            idx_list = np.linspace(0, len(q_l) - 1, num=int(len(q_l) / k), endpoint=True, dtype=int)
            key_times = [t_list[i] for i in idx_list]
            key_rots = R.from_quat([q_l[i].as_quat() for i in idx_list])

            slerp = Slerp(key_times, key_rots)

            idx_list = np.linspace(0, len(q_l) - 1, num=int(len(q_l)), endpoint=True, dtype=int)
            key_times = [t_list[i] for i in idx_list]

            q_interp = slerp(key_times)
            q_smooth.append([q_interp[i] for i in range(len(q_interp))])

    return q_smooth


def _filter(p_list, q_list, t_list):
    """   Extract a smooth velocity profile (non-zero except near the attractor)  """

    min_thold = 0.05
    pct_thold = 0.8

    L = len(q_list)

    p_filter = []
    q_filter = []
    t_filter = []

    for l in range(L):
        M = len(q_list[l])
        M_thold = M * pct_thold

        p_filter_l = [p_list[l][0, :]]
        q_filter_l = [q_list[l][0]]
        t_filter_l = [t_list[l][0]]

        for i in range(M - 1):
            q_i = q_filter_l[-1]
            q_ip1 = q_list[l][i + 1]
            dt = t_list[l][i + 1] - t_list[l][i]
            w = _compute_ang_vel(q_i, q_ip1, dt)

            if (i <= M_thold and np.linalg.norm(w) >= min_thold) or i > M_thold:
                p_filter_l.append(p_list[l][i + 1, :])
                q_filter_l.append(q_list[l][i + 1])
                t_filter_l.append(t_list[l][i + 1])

        p_filter.append(np.array(p_filter_l))
        q_filter.append(q_filter_l)
        t_filter.append(np.array(t_filter_l))

    return p_filter, q_filter, t_filter


def pre_process(p_raw, q_raw, t_raw, opt="savgol"):
    p_in, p_att = _shift_pos(p_raw)
    q_in, q_att = _shift_ori(q_raw)

    # q_in                    = _smooth_ori(q_in, q_att, opt)  # needed or not?

    # p_in, q_in, t_in        = _filter(p_in, q_in, t_raw)  # needed or not?

    return p_in, q_in, t_raw


# def pre_process(p_raw, q_raw, t_raw, opt="savgol"):
#     # 将输入的四元数数组转换为 Rotation 对象
#     q_raw = [R.from_quat(q) for q in q_raw]
#     p_in, p_att = _shift_pos(p_raw)
#     q_in, q_att = _shift_ori(q_raw)
#     # q_in                    = _smooth_ori(q_in, q_att, opt)  # needed or not?
#     # p_in, q_in, t_in        = _filter(p_in, q_in, t_raw)  # needed or not?
#     return p_in, q_in, t_raw


import numpy as np


def compute_output(p_list, q_list, t_list):
    L = len(q_list)  # 轨迹的数量

    p_out = []  # 速度向量的输出列表
    q_out = []  # 四元数的输出列表

    for l in range(L):  # 遍历每条轨迹
        M = len(q_list[l])  # 轨迹中的位置点数量

        p_out_l = []  # 当前轨迹的速度向量列表
        q_out_l = []  # 当前轨迹的四元数列表

        for i in range(M - 1):  # 遍历每个位置点
            p_i = p_list[l][i, :]  # 当前点位置
            p_ip1 = p_list[l][i + 1,:]  # 下一个点位置

            q_i = q_list[l][i]  # 当前点四元数
            q_ip1 = q_list[l][i + 1]  # 下一个点四元数

            dt = t_list[l][i + 1] - t_list[l][i]  # 当前点和下一个点之间的时间间隔

            v = (p_ip1 - p_i) / dt  # 计算速度向量

            p_out_l.append(v)  # 添加速度向量到当前轨迹的列表
            q_out_l.append(q_ip1)  # 添加四元数到当前轨迹的列表

        p_out_l.append(v)  # 对于最后一个位置点，复制前一个速度向量
        q_out_l.append(q_ip1)  # 对于最后一个位置点，复制前一个四元数

        p_out.append(np.array(p_out_l))  # 将当前轨迹的速度向量列表转换为数组并添加到输出列表
        q_out.append(q_out_l)  # 将当前轨迹的四元数列表添加到输出列表

    return p_out, q_out  # 返回计算结果


def extract_state(p_list, q_list):
    L = len(q_list)  # 轨迹数量

    p_init = []  # 每条轨迹的初始位置列表
    q_init = []  # 每条轨迹的初始四元数列表

    for l in range(L):  # 遍历每条轨迹
        p_init.append(p_list[l][0, :])  # 提取第 l 条轨迹的初始位置
        q_init.append(q_list[l][0])  # 提取第 l 条轨迹的初始四元数

    p_att = p_list[0][-1, :]  # 提取第一条轨迹的最终位置
    q_att = q_list[0][-1]  # 提取第一条轨迹的最终四元数

    return p_init, q_init, p_att, q_att  # 返回提取的状态


def rollout_list(p_in, q_in, p_out, q_out):
    """ Roll out the nested list into a single list of M entries """

    L = len(q_in)  # 轨迹数量

    for l in range(L):
        if l == 0:
            # 对于第一个轨迹，直接赋值
            p_in_rollout = p_in[l]
            q_in_rollout = q_in[l]
            p_out_rollout = p_out[l]
            q_out_rollout = q_out[l]
        else:
            # 对于后续的轨迹，将数据拼接到第一个轨迹的末尾
            p_in_rollout = np.vstack((p_in_rollout, p_in[l]))
            q_in_rollout += q_in[l]
            p_out_rollout = np.vstack((p_out_rollout, p_out[l]))
            q_out_rollout += q_out[l]

    return p_in_rollout, q_in_rollout, p_out_rollout, q_out_rollout
