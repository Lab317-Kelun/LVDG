import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.transform import Rotation as R
from .quat_tools import *
import random


# font = {'family' : 'Times New Roman',
#          'size'   : 18
#          }
# mpl.rc('font', **font)



def plot_omega(omega_test):

    # omega_test = np.vstack(omega_test)
    M, N = omega_test.shape
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(3):
        axs[k].scatter(np.arange(M), omega_test[:, k], s=5, color=colors[k])
        # axs[k].set_ylim([0, 1])



def plot_gamma(gamma_arr, **argv):

    M, K = gamma_arr.shape

    fig, axs = plt.subplots(K, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(K):
        axs[k].scatter(np.arange(M), gamma_arr[:, k], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
    
    if "title" in argv:
        axs[0].set_title(argv["title"])
    else:
        axs[0].set_title(r"$\gamma(\cdot)$ over Time")


import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.ticker import MaxNLocator


def plot_gmm(p_in, gmm):
    # 提取聚类标签和高斯混合模型中的高斯分布个数
    label = gmm.assignment_arr
    K = gmm.K

    # 定义颜色列表，用于不同聚类的点
    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    # 根据聚类标签映射颜色
    color_mapping = np.take(colors, label)

    # 创建一个3D图形对象
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    # 绘制输入点，使用映射的颜色
    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    # 定义颜色列表，用于高斯分布的主轴
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

    # 获取x轴的范围，并设置缩放比例
    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min) / 4

    # 循环遍历每个高斯分布
    for k in range(K):
        label_k = np.where(label == k)[0]
        p_in_k = p_in[label_k, :]
        loc = np.mean(p_in_k, axis=0)

        r = gmm.gaussian_list[k]["mu"]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)

    # 设置图形的轴和标签
    ax.axis('equal')
    ax.set_xlabel('r(m)', labelpad=20)
    ax.set_ylabel('p(m)', labelpad=20)
    ax.set_zlabel('y(m)', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


