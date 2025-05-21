import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from cvxopt import solvers, matrix

np.random.seed(5)


class LearnSds:
    def __init__(self, lf_learner, ods_learner):
        '''
        Initializing the Stable ADS
        '''
        self.lf_learner = lf_learner
        self.ods_learner = ods_learner
        self.d_x = lf_learner.d_x

    def predict(self, energy_function_parameter, x, func_rho, P=None, r_thres=0.0, eta=0.1):
        '''
        Prediction of the Stable ADS.
        Solving the Optimization problem described by (46) and (47)
        in the paper
        '''
        d_x = np.shape(x)[0]
        if P is None:
            P = np.eye(d_x)
        x_P_norm = np.sqrt(np.dot(np.dot(P, x), x))
        if x_P_norm == 0.0:
            dot_x, u = np.zeros(self.lf_learner.d_x), np.zeros(self.lf_learner.d_x)
            return dot_x, u
        elif ((x_P_norm**2) > (r_thres**2) * ((1 - eta)**2)) and ((x_P_norm**2) < (r_thres**2) * ((1 + eta)**2)):
            # print('in QP case')
            P_ = matrix(np.eye(d_x))
            q = matrix(np.zeros(d_x))
            dvdx = self.lf_learner.dvdx(energy_function_parameter, x)
            G = matrix(np.array([dvdx, np.dot(P, x)]))
            ods_dot_x = self.ods_learner.predict(x.reshape(1, 3)).reshape(-1)
            rho = func_rho(x)
            temp = np.dot(ods_dot_x, dvdx) + rho
            h0 = -temp
            h1 = -np.dot(np.dot(P, ods_dot_x), x)
            h = matrix([h0, h1])
            solvers.options['show_progress'] = False
            solution = solvers.qp(P_, q, G, h)
            u = np.array(solution['x']).reshape(-1)
            dot_x = ods_dot_x + u
            return dot_x, u
        else:
            dvdx = self.lf_learner.dvdx(energy_function_parameter, x)
            dvdx_norm_2 = np.dot(dvdx, dvdx)
            ods_dot_x = self.ods_learner.predict(x.reshape(1, 3)).reshape(-1)
            rho = func_rho(x)
            temp = np.dot(ods_dot_x, dvdx) + rho
            if temp > 0:
                u = -temp / dvdx_norm_2 * dvdx
                dot_x = ods_dot_x + u
            else:
                u = np.zeros(self.lf_learner.d_x)
                dot_x = ods_dot_x
            return dot_x, u

    def func_rho(self, x):
        '''
        A default function rho(x), see Eq. (57)
        in the paper
        '''
        gamma = 10.0
        x_norm_2 = np.dot(x, x)
        beta = (self.lf_learner.overline_x ** 2)
        max_v_in_set = np.max(np.sqrt(np.sum(self.lf_learner.dot_x_set ** 2, axis=1)))
        scale = max_v_in_set / gamma
        return scale * (1 - np.exp(-0.5 * x_norm_2 / beta))

    def energy_function(self, x, lf_parameter, func_rho, P=None, r_thres=0.0, eta=0.1):
        '''
        A default energy function for the Stable ADS.
        You need to implement this function based on your specific requirements.
        '''
        # Example implementation (you need to modify this based on your specific energy function)
        energy = np.dot(x, x) / 2  # Simple quadratic energy function
        return energy

    def plot_energy_field(self, lf_parameter, func_rho=None, P=None, r_thres=0.0, eta=0.1, area_Cartesian=None):
        if func_rho is None:
            func_rho = self.func_rho

        if area_Cartesian is None:
            x_1_min = np.min(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 0])
            x_1_max = np.max(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 0])
            x_2_min = np.min(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 1])
            x_2_max = np.max(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 1])
            x_3_min = np.min(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 2])
            x_3_max = np.max(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 2])

            delta_x1 = x_1_max - x_1_min
            x_1_min = x_1_min - 0.2 * delta_x1
            x_1_max = x_1_max + 0.2 * delta_x1
            delta_x2 = x_2_max - x_2_min
            x_2_min = x_2_min - 0.2 * delta_x2
            x_2_max = x_2_max + 0.2 * delta_x2
            delta_x3 = x_3_max - x_3_min
            x_3_min = x_3_min - 0.2 * delta_x3
            x_3_max = x_3_max + 0.2 * delta_x3

            num = 20  # 3D 网格点数
            step = np.min(np.array([(x_1_max - x_1_min) / num, (x_2_max - x_2_min) / num, (x_3_max - x_3_min) / num]))
            area_Cartesian = {'x_1_min': x_1_min, 'x_1_max': x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max,
                              'x_3_min': x_3_min, 'x_3_max': x_3_max, 'step': step}

        area = area_Cartesian
        step = area['step']
        x1 = np.arange(area['x_1_min'], area['x_1_max'], step)
        x2 = np.arange(area['x_2_min'], area['x_2_max'], step)
        x3 = np.arange(area['x_3_min'], area['x_3_max'], step)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X1, X2, X3 = np.meshgrid(x1, x2, x3)
        V = np.zeros_like(X1)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                for k in range(X1.shape[2]):
                    x = np.array([X1[i, j, k], X2[i, j, k], X3[i, j, k]])
                    V[i, j, k] = self.energy_function(x, lf_parameter, func_rho, P=P, r_thres=r_thres, eta=eta)

        ax.plot_surface(X1[:, :, 0], X2[:, :, 0], V[:, :, int(V.shape[2] / 2)], cmap='viridis')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Energy')
        plt.show()

    def plot_trajectory(self, lf_parameter, func_rho=None, P=None, r_thres=0.0, eta=0.1):
        if func_rho is None:
            func_rho = self.func_rho

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mark_size = 50
        ax.scatter(0, 0, 0, c='black', alpha=1.0, s=mark_size, marker='X')
        ax.scatter(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 0], self.lf_learner.x_set.reshape(-1, self.d_x)[:, 1],
                   self.lf_learner.x_set.reshape(-1, self.d_x)[:, 2], c='red', alpha=1.0, s=mark_size, marker='o')
        n_tra = np.shape(self.lf_learner.x_set)[0]
        for i in range(n_tra):
            ax.scatter(self.lf_learner.x_set[i, 0, 0], self.lf_learner.x_set[i, 0, 1], self.lf_learner.x_set[i, 0, 2],
                       c='black', alpha=1.0, s=mark_size, marker='o')

        # 用黑色线连接红色点
        for traj in self.lf_learner.x_set:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='black', linewidth=2, alpha=1.0)

        x0s = self.lf_learner.x_set[:, 0, :]
        for i in range(len(x0s)):
            self.plot_repro(lf_parameter, x0s[i, :], func_rho=func_rho, original_trajectory=self.lf_learner.x_set[i],
                            P=P, r_thres=r_thres, eta=eta)

        plt.show()

    def plot_repro(self, lf_parameter, x0, func_rho, original_trajectory, P=None, r_thres=0.0, eta=0.1):
        x = x0
        period = 1e-2
        steps = int(100 / period)
        x_tra = [x]

        for i in range(steps):
            desired_v, u = self.predict(lf_parameter, x, func_rho, P=P, r_thres=r_thres, eta=eta)
            print(f"Step {i} - desired_v: {desired_v}, u: {u}")
            if np.all(desired_v == 0):
                print(f"At step {i}, desired_v is zero.")
            x = x + desired_v * period
            x_tra.append(x)

        x_tra = np.array(x_tra)

        print("Original trajectory:", original_trajectory)
        print("Predicted trajectory:", x_tra)

        # 绘制原始轨迹
        fig_original = plt.figure()
        ax_original = fig_original.add_subplot(111, projection='3d')
        ax_original.plot(original_trajectory[:, 0], original_trajectory[:, 1], original_trajectory[:, 2], c='blue',
                         linewidth=2, alpha=1.0)
        ax_original.set_title("Original Trajectory")
        plt.show()

        # 绘制预测轨迹
        fig_predicted = plt.figure()
        ax_predicted = fig_predicted.add_subplot(111, projection='3d')
        ax_predicted.plot(x_tra[:, 0], x_tra[:, 1], x_tra[:, 2], c='black', linewidth=2, alpha=1.0)
        ax_predicted.set_title("Predicted Trajectory")
        plt.show()