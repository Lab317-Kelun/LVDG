'''
Learning the neural energy function "NEUM" .
The second learning approach described in the paper is used.
Details can be found in the paper:
"Learning a flexible neural energy function with a unique minimum
for stable and accurate demonstration learning"
'''

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
np.random.seed(5)


class LearnNeum:
    def __init__(self, manually_design_set, d_H, L_2 = 1e-6, epsilon=1e-10, alpha=1e-10):
        self.x_set, self.dot_x_set, self.t_set = manually_design_set
        self.d_H = d_H
        self.d_x = np.shape(self.x_set)[2]
        self.parameter_length = self.d_H * self.d_x + 3 * self.d_H
        self.data_size = np.shape(self.x_set)[0] * np.shape(self.x_set)[1]
        self.epsilon = epsilon
        self.overline_x = np.max(np.sqrt(np.sum(self.x_set.reshape(-1, self.d_x) ** 2, axis=1)))
        self.scale = (self.overline_x / 2)**(1 + self.epsilon)
        self.alpha = alpha  # weight for the quadratic term
        self.L2 = L_2  # L2 regularization parameter

    def v(self, parameters, x):
        '''
        Compute the neural energy function value given by
        the parameters and position x
        '''
        def forward_prop(parameters, x):
            '''
            The forward propagation of the
            neural network
            '''
            A = parameters[0: (self.d_x + 1) * self.d_H].reshape(self.d_H, self.d_x + 1)
            b = parameters[((self.d_x + 1) * self.d_H): ((self.d_x + 2) * self.d_H)]
            w = parameters[((self.d_x + 2) * self.d_H): ((self.d_x + 3) * self.d_H)]
            x_norm = np.sqrt(np.dot(x, x))
            z = np.hstack((np.array([x_norm**(1.0 + self.epsilon)]), (x_norm**self.epsilon) * x)) / self.scale
            f = np.tanh(np.dot(A, z) + b)
            output = np.dot(w, f)
            return output

        v = forward_prop(parameters, x)
        v_0 = forward_prop(parameters, np.zeros(self.d_x))
        return v + self.alpha * np.dot(x, x) - v_0

    def dvdx(self, parameters, x):
        '''
        Compute the gradient of neural energy
        function v(x) with respect to x
        '''
        x_norm = np.sqrt(np.dot(x, x))
        if x_norm == 0.0:
            return np.zeros(self.d_x)
        else:
            A = parameters[0: (self.d_x + 1) * self.d_H].reshape(self.d_H, self.d_x + 1)
            b = parameters[((self.d_x + 1) * self.d_H): ((self.d_x + 2) * self.d_H)]
            w = parameters[((self.d_x + 2) * self.d_H): ((self.d_x + 3) * self.d_H)]
            x_norm = np.sqrt(np.dot(x, x))
            z = np.hstack((np.array([x_norm ** (1.0 + self.epsilon)]), (x_norm ** self.epsilon) * x)) / self.scale
            f = np.tanh(np.dot(A, z) + b)
            dvdf = w
            dfdz = (1 - f**2).reshape(-1, 1) * A
            temp1 = ((1 + self.epsilon) * (x_norm ** self.epsilon) * (x / x_norm)).reshape(1, -1) / self.scale
            temp2 = (self.epsilon * (x_norm ** self.epsilon) * np.dot((x / x_norm).reshape(-1, 1), (x / x_norm).reshape(1, -1)) + (x_norm ** self.epsilon) * np.eye(self.d_x)) / self.scale
            dzdx = np.vstack((temp1, temp2))
            dvdx = (np.dot(np.dot(dvdf.reshape(1, -1), dfdz), dzdx)).reshape(-1)
            return dvdx + 2 * self.alpha * x

    def ddvdxdp(self, parameters, x):
        '''
        Compute the Jacobian ddvdxdp
        '''
        x_norm = np.sqrt(np.dot(x, x))
        if x_norm == 0.0:
            return np.zeros(self.d_x, self.parameter_length)
        else:
            A = parameters[0: (self.d_x + 1) * self.d_H].reshape(self.d_H, self.d_x + 1)
            b = parameters[((self.d_x + 1) * self.d_H): ((self.d_x + 2) * self.d_H)]
            w = parameters[((self.d_x + 2) * self.d_H): ((self.d_x + 3) * self.d_H)]
            x_norm = np.sqrt(np.dot(x, x))
            z = np.hstack((np.array([x_norm ** (1.0 + self.epsilon)]), (x_norm ** self.epsilon) * x)) / self.scale
            f = np.tanh(np.dot(A, z) + b)
            temp1 = ((1 + self.epsilon) * (x_norm ** self.epsilon) * (x / x_norm)).reshape(1, -1) / self.scale
            temp2 = (self.epsilon * (x_norm ** self.epsilon) * np.dot((x / x_norm).reshape(-1, 1), (x / x_norm).reshape(1, -1)) + (x_norm ** self.epsilon) * np.eye(self.d_x)) / self.scale
            dzdx = np.vstack((temp1, temp2))

            temp3 = (w * (1 - (f**2))).reshape(self.d_H, 1, 1) * np.expand_dims(np.eye(self.d_x + 1), 0)
            temp4 = (w * (-2 * f) * (1 - (f**2))).reshape(self.d_H, 1, 1) * np.dot(np.expand_dims(A, 2), z.reshape(1, -1))
            ddvdxdA = np.dot(dzdx.T, temp3 + temp4).reshape(self.d_x, -1)

            ddvdxdb = np.dot(dzdx.T, ((w * (-2 * f) * (1 - (f**2))).reshape(self.d_H, 1) * A).T)
            ddvdxdw = np.dot(dzdx.T, ((1 - (f**2)).reshape(self.d_H, 1) * A).T)
            ddvdxdp = np.hstack((ddvdxdA, ddvdxdb, ddvdxdw))
            return ddvdxdp

    def obj_and_gradient_for_single_pair_points(self, parameters, x, dot_x, beta):
        '''
        Compute the obj function and parameter-gradient
        on a data-pair (x_k, dot_x_k)
        '''
        dvdx = self.dvdx(parameters, x)
        dvdx_norm = np.sqrt(np.dot(dvdx, dvdx))
        dot_x_norm = np.sqrt(np.dot(dot_x, dot_x))
        if dvdx_norm == 0 or dot_x_norm == 0:
            return None, None
        else:
            overline_j = np.dot(dvdx, dot_x) / np.sqrt(np.dot(dvdx, dvdx) * np.dot(dot_x, dot_x))
            djd_dvdx = (dot_x / dot_x_norm).reshape(1, -1).dot(
                dvdx_norm * np.eye(self.d_x) - dvdx.reshape(-1, 1).dot(dvdx.reshape(1, -1)) / dvdx_norm) / (
                                   dvdx_norm ** 2)
            ddvdxdp = self.ddvdxdp(parameters, x)

            overline_djdp = np.dot(djd_dvdx, ddvdxdp).reshape(-1)
            jk = np.tanh(beta * overline_j)
            djkdp = (1 - (np.tanh(beta * overline_j) ** 2)) * beta * overline_djdp
            return jk, djkdp

    def obj_and_gradient_for_data_set(self, parameters, beta):
        '''
        Compute the objective function on the demonstration set
        '''
        J = 0
        dJdp = np.zeros(self.parameter_length)
        number_unsingular_points = 0
        x_set = self.x_set.reshape(-1, self.d_x)
        dot_x_set = self.dot_x_set.reshape(-1, self.d_x)
        for k in range(self.data_size):
            x, dot_x = x_set[k, :], dot_x_set[k, :]
            jk, djkdp = self.obj_and_gradient_for_single_pair_points(parameters, x, dot_x, beta)
            if jk is not None:
                number_unsingular_points = number_unsingular_points + 1
                J = J + jk
                dJdp = dJdp + djkdp
        J = J / number_unsingular_points + self.L2 * np.sum(parameters**2)
        dJdp = dJdp / number_unsingular_points + 2 * self.L2 * parameters
        return J, dJdp

    def train(self, save_path=None, beta=10.0, maxiter=500):
        '''
        Training the neural energy function
        '''
        def cons_f(parameters):
            '''
            Compute the nonlinear constraints
            '''
            A = parameters[0: (self.d_x + 1) * self.d_H].reshape(self.d_H, self.d_x + 1)
            cons_f = A[:, 0] ** 2 - np.sum(A[:, 1:]**2, axis=1)
            return cons_f

        def cons_jac(parameters):
            '''
            Compute the Jacobian of the constrained functions
            '''
            A_ = 2 * parameters[0: (self.d_x + 1) * self.d_H].reshape(self.d_H, self.d_x + 1)
            A_[:, 1:] = -A_[:, 1:]
            temp1 = np.kron(np.eye(self.d_H), A_)
            indices = np.arange(0, self.d_H ** 2, self.d_H + 1)
            dcdp = np.hstack((temp1[indices, :], np.zeros((self.d_H, self.parameter_length - (self.d_x + 1) * self.d_H))))
            return dcdp

        def cons_hvp(parameters, v):
            '''
            Compute the Hessian-vector-product of the constrained functions
            '''
            hvp = np.zeros((self.parameter_length, self.parameter_length))
            ddcidakak = np.eye(self.d_x + 1) * (-2.0)
            ddcidakak[0, 0] = 2.0
            for i in np.arange(self.d_H):
                ddcidpdp = np.zeros((self.parameter_length, self.parameter_length))
                ddcidpdp[(i * (self.d_x + 1)):((i + 1) * (self.d_x + 1)), (i * (self.d_x + 1)):((i + 1) * (self.d_x + 1))] = ddcidakak
                hvp = hvp + v[i] * ddcidpdp
            return hvp

        nonlinear_constraint = NonlinearConstraint(cons_f, 1e-8, 1e0, jac=cons_jac, hess=cons_hvp)

        # Setting lower and upper bounds for parameters
        paras_lb = np.ones(self.parameter_length) * (-np.inf)
        paras_lb[0:((self.d_x + 1) * self.d_H):(self.d_x + 1)] = 1e-8  # For A[:, 0]
        paras_lb[((self.d_x + 2) * self.d_H): ((self.d_x + 3) * self.d_H)] = 1e-8  # For w
        paras_ub = np.ones(self.parameter_length) * np.inf
        bounds = Bounds(paras_lb, paras_ub)

        # Initialize the parameters
        parameters = np.random.uniform(-0.5, 0.5, self.parameter_length)
        # Optimizing
        res = minimize(self.obj_and_gradient_for_data_set, parameters, method='trust-constr', jac=True, args=(beta),
                       options={'disp': True, 'maxiter': maxiter, 'xtol': 1e-50, 'gtol': 1e-20}, bounds=bounds,
                       constraints=[nonlinear_constraint], callback=self.callback)
        parameters = res.x
        if save_path is not None:
            np.savetxt(save_path, parameters)
        return parameters

    def callback(self, parameters, state):
        '''
        Show learning details during the training process
        '''
        if state.nit % 50 == 0 or state.nit == 1:
            obj_l2 = self.L2 * np.sum(parameters ** 2)
            obj_cost = state.fun - obj_l2

            print('---------------------------------- iter ', state.nit, '----------------------------------')
            print('running time: ', state.execution_time)
            print('obj_cost: ', obj_cost, 'L2_cost: ', obj_l2)
            print('maximum constr_violation: ', state.constr_violation)

        '''
        # Save figs
        if state.nit % 5 == 0 or state.nit == 1:
            texts ="Iteration: " + str(state.nit) + "\nConstr_violation: " + str(np.round(state.constr_violation, 2))
            save_path = 'NEUM_figs/Sshape/fig' + str(int(state.nit / 5)) + '.jpg'
            save_info = [save_path, texts]
            import matplotlib.pyplot as plt
            plot_handle = plt
            plot_handle.clf()
            self.show_learning_result(parameters, plot_handle=plot_handle, save_info=save_info)
        '''
    def show_learning_result(self, parameters, num_levels=10, scatter_flag=True, plot_handle=None, area_Cartesian=None, save_info=None):
        '''
        Plotting the learned neural energy function
        '''
        if self.d_x == 2:
            if area_Cartesian is None:
                x_1_min = np.min(self.x_set.reshape(-1, self.d_x)[:, 0])
                x_1_max = np.max(self.x_set.reshape(-1, self.d_x)[:, 0])
                x_2_min = np.min(self.x_set.reshape(-1, self.d_x)[:, 1])
                x_2_max = np.max(self.x_set.reshape(-1, self.d_x)[:, 1])

                delta_x1 = x_1_max - x_1_min
                x_1_min = x_1_min - 0.1 * delta_x1
                x_1_max = x_1_max + 0.1 * delta_x1
                delta_x2 = x_2_max - x_2_min
                x_2_min = x_2_min - 0.1 * delta_x2
                x_2_max = x_2_max + 0.1 * delta_x2

                num = 80
                step = np.min(np.array([(x_1_max - x_1_min) / num, (x_2_max - x_2_min) / num]))

                area_Cartesian = {'x_1_min': x_1_min, 'x_1_max': x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max, 'step': step}
            area = area_Cartesian
            step = area['step']
            x1 = np.arange(area['x_1_min'], area['x_1_max'], step)
            x2 = np.arange(area['x_2_min'], area['x_2_max'], step)
            length_x1 = np.shape(x1)[0]
            length_x2 = np.shape(x2)[0]
            X1, X2 = np.meshgrid(x1, x2)
            V = np.zeros((length_x2, length_x1))
            V_list = np.zeros(length_x2 * length_x1)

            for i in range(length_x2):
                for j in range(length_x1):
                    x = np.array([x1[j], x2[i]])
                    V[i, j] = self.v(parameters, x)
                    V_list[i * length_x1 + j] = V[i, j]
            levels = np.sort(V_list, axis=0)[0::int(length_x2 * length_x1 / num_levels)]
            levels_ = []
            for i in range(np.shape(levels)[0]):
                if i is 0:
                    levels_.append(levels[i])
                else:
                    if levels[i] != levels[i - 1]:
                        levels_.append(levels[i])
            levels_ = np.array(levels_)
            # If we use external plot handle,
            # the function plt.show() will not ba called
            show_flag = False
            if plot_handle is None:
                import matplotlib.pyplot as plt
                plot_handle = plt
                show_flag = True
            contour = plot_handle.contour(X1, X2, V, levels=levels_, alpha=0.8, linewidths=1.0)
            # Show lf values on contours
            plot_handle.clabel(contour, fontsize=8)
            mark_size = 2
            if scatter_flag is True:
                plot_handle.scatter(0, 0, c='black', alpha=1.0, s=50, marker='X')
                n_tras = np.shape(self.x_set)[0]
                for tra_k in range(n_tras):
                    length_tra_k = np.shape(self.x_set[tra_k, :, :])[0]
                    for i in np.arange(length_tra_k - 1):
                        if i == 0:
                            plot_handle.scatter(self.x_set[tra_k, i, 0], self.x_set[tra_k, i, 1], c='black', alpha=1.0, s=15, marker='o')
                        elif self.v(parameters, self.x_set[tra_k, i + 1, :]) >= self.v(parameters, self.x_set[tra_k, i, :]):
                            plot_handle.scatter(self.x_set[tra_k, i, 0], self.x_set[tra_k, i, 1], c='blue', alpha=1.0, s=mark_size, marker='x')
                        else:
                            plot_handle.scatter(self.x_set[tra_k, i, 0], self.x_set[tra_k, i, 1], c='red', alpha=1.0, s=mark_size, marker='o')
            if save_info is not None:
                show_flag = False
                text_position = np.array([x_1_min + delta_x1 * 0.03, x_2_max - delta_x2 * 0.16])
                save_path, text_info = save_info
                plot_handle.text(text_position[0], text_position[1], text_info, fontsize=12, color='black')
                plot_handle.savefig(save_path)
            if show_flag is True:
                plot_handle.show()
        elif self.d_x == 3:
            if area_Cartesian is None:
                x_1_min = np.min(self.x_set.reshape(-1, self.d_x)[:, 0])
                x_1_max = np.max(self.x_set.reshape(-1, self.d_x)[:, 0])
                x_2_min = np.min(self.x_set.reshape(-1, self.d_x)[:, 1])
                x_2_max = np.max(self.x_set.reshape(-1, self.d_x)[:, 1])
                x_3_min = np.min(self.x_set.reshape(-1, self.d_x)[:, 2])
                x_3_max = np.max(self.x_set.reshape(-1, self.d_x)[:, 2])
                # enlarge the area
                delta_x1 = x_1_max - x_1_min
                x_1_min = x_1_min - 0.1 * delta_x1
                x_1_max = x_1_max + 0.1 * delta_x1
                delta_x2 = x_2_max - x_2_min
                x_2_min = x_2_min - 0.1 * delta_x2
                x_2_max = x_2_max + 0.1 * delta_x2
                delta_x3 = x_3_max - x_3_min
                x_3_min = x_3_min - 0.1 * delta_x3
                x_3_max = x_3_max + 0.1 * delta_x3

                num = 80
                step = np.min(np.array([(x_1_max - x_1_min) / num, (x_2_max - x_2_min) / num]))
                area_Cartesian = {'x_1_min': x_1_min, 'x_1_max': x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max,
                                  'x_3_min': x_3_min, 'x_3_max': x_3_max, 'step': step}
                #print(area_Cartesian)

            area = area_Cartesian
            step = area['step']
            x1 = np.arange(area['x_1_min'], area['x_1_max'], step)
            x2 = np.arange(area['x_2_min'], area['x_2_max'], step)
            if area['x_3_min'] == area['x_3_max']:
                x3 = np.array([area['x_3_min']])
            else:
                x3 = np.arange(area['x_3_min'], area['x_3_max'], step)
            #print(x1)
            #print(x3)
            length_x1 = np.shape(x1)[0]
            length_x2 = np.shape(x2)[0]
            length_x3 = np.shape(x3)[0]

            #print(f"length_x1: {length_x1}, length_x2: {length_x2}, length_x3: {length_x3}")

            # 确保 x1, x2, x3 都不是空的
            if length_x1 == 0 or length_x2 == 0 or length_x3 == 0:
                raise ValueError(
                    "One of the generated arrays x1, x2, x3 is empty. Please check the step size and the range.")

            V = np.zeros((length_x1, length_x2, length_x3))
            X = []
            for i in range(length_x1):
                for j in range(length_x2):
                    for k in range(length_x3):
                        x = np.array([x1[i], x2[j], x3[k]])
                        X.append(x)
                        V[i, j, k] = self.v(parameters, x)
            X = np.array(X)
            # 调试信息: 打印形状
            print(f"Shape of X: {X.shape}")
            # plotting isosurfaces
            show_flag = False
            if plot_handle is None:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                plot_handle = fig.add_subplot(111, projection='3d')
                show_flag = True
            from skimage.measure import marching_cubes
            number_isosurfaces = 5
            interested_tra_index = 4
            length_tra = np.shape(self.x_set[interested_tra_index, :, :])[0]
            gap = int(length_tra / number_isosurfaces)
            for i in range(number_isosurfaces):
                x_iso_expected = self.x_set[interested_tra_index, i * gap, :]
                # 调试信息: 打印形状
                print(f"Shape of x_iso_expected before reshape: {x_iso_expected.shape}")
                x_iso_expected = x_iso_expected.reshape(1, -1)
                print(f"Shape of x_iso_expected after reshape: {x_iso_expected.shape}")
                diff = np.sum((X - x_iso_expected) ** 2, axis=1)
                x_iso = X[np.argmin(diff), :]
                plot_handle.scatter3D(x_iso[0], x_iso[1], x_iso[2], c='blue', alpha=1.0, s=30, marker='D')
                v_iso = self.v(parameters, x_iso)
                verts, faces, _, _ = marching_cubes(V, v_iso, spacing=(step, step, step))
                # compensate the offset
                verts = verts + np.array([x_1_min, x_2_min, x_3_min]).reshape(1, -1)
                plot_handle.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1,
                                         alpha=0.5)
            if scatter_flag is True:
                mark_size = 5
                plot_handle.scatter3D(0, 0, 0, c='black', alpha=1.0, s=50, marker='X')
                n_tras = np.shape(self.x_set)[0]
                for tra_k in range(n_tras):
                    length_tra_k = np.shape(self.x_set[tra_k, :, :])[0]
                    for i in np.arange(length_tra_k - 1):
                        if i == 0:
                            plot_handle.scatter3D(self.x_set[tra_k, i, 0], self.x_set[tra_k, i, 1],
                                                  self.x_set[tra_k, i, 2], c='black', alpha=1.0, s=15, marker='o')
                        elif self.v(parameters, self.x_set[tra_k, i + 1, :]) >= self.v(parameters,
                                                                                       self.x_set[tra_k, i, :]):
                            plot_handle.scatter3D(self.x_set[tra_k, i, 0], self.x_set[tra_k, i, 1],
                                                  self.x_set[tra_k, i, 2], c='blue', alpha=1.0, s=mark_size,
                                                  marker='x')
                        else:
                            plot_handle.scatter3D(self.x_set[tra_k, i, 0], self.x_set[tra_k, i, 1],
                                                  self.x_set[tra_k, i, 2], c='red', alpha=1.0, s=mark_size,
                                                  marker='o')
            if show_flag is True:
                plt.show()
        return















