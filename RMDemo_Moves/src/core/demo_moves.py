import sys
import os
import numpy as np

# 将src的父目录添加到sys.path，以便能够导入src中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from RMDemo_Moves.src.Robotic_Arm.rm_robot_interface  import *


class RobotArmController:
    def __init__(self, ip, port, level=3, mode=2):
        """
        初始化并连接到机械臂。

        参数:
            ip (str): 机械臂的IP地址。
            port (int): 端口号。
            level (int, 可选): 连接级别。默认为3。
            mode (int, 可选): 线程模式(0: 单线程, 1: 双线程, 2: 三线程)。默认为2。
        """
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.handle.id == -1:
            print("\n连接机械臂失败\n")
            exit(1)
        else:
            print(f"\n成功连接到机械臂: {self.handle.id}\n")

    def disconnect(self):
        """
        断开与机械臂的连接。

        返回:
            None
        """
        handle = self.robot.rm_delete_robot_arm()
        if handle == 0:
            print("\n成功断开与机械臂的连接\n")
        else:
            print("\n断开与机械臂的连接失败\n")

    def movej(self, joint, v=10, r=0, connect=0, block=1):
        """
        执行movej运动。

        参数:
            joint (list of float): 关节位置。
            v (float, 可选): 运动速度。默认为20。
            connect (int, 可选): 轨迹连接标志。默认为0。
            block (int, 可选): 函数是否阻塞（1为阻塞，0为非阻塞）。默认为1。
            r (float, 可选): 混合半径。默认为0。

        返回:
            None
        """
        movej_result = self.robot.rm_movej(joint, v, r, connect, block)
        if movej_result == 0:
            print("\nmovej运动成功\n")
        else:
            print("\nmovej运动失败，错误码: ", movej_result, "\n")

    def movej_p(self, pose, v=10, r=0, connect=0, block=0):
        """
        执行movej_p运动。

        参数:
            pose (list of float): 位置 [x, y, z, rx, ry, rz]。
            v (float, 可选): 运动速度。默认为20。
            connect (int, 可选): 轨迹连接标志。默认为0。
            block (int, 可选): 函数是否阻塞（1为阻塞，0为非阻塞）。默认为1。
            r (float, 可选): 混合半径。默认为0。

        返回:
            None
        """
        movej_p_result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if movej_p_result == 0:
            print("\nmovej_p运动成功\n")
        else:
            print("\nmovej_p运动失败，错误码: ", movej_p_result, "\n")

    def moves(self, move_positions=None, speed=10, blending_radius=0, block=0):
        """
        执行一系列的移动操作。

        参数:
            move_positions (list of float, 可选): 要移动到的位置列表，每个位置为 [x, y, z, rx, ry, rz]。
            speed (int, 可选): 移动速度。默认为20。
            block (int, 可选): 函数是否阻塞（1为阻塞，0为非阻塞）。默认为1。
            blending_radius (float, 可选): 移动的混合半径。默认为0。

        返回:
            None
        """
        if move_positions is None:
            move_positions = [
                [-0.205286,0.016923,0.419555,3.142,0.209,-0.084]

            ]

        for i, pos in enumerate(move_positions):
            current_connect = 1 if i < len(move_positions) - 1 else 0
            moves_result = self.robot.rm_moves(pos, speed, blending_radius, current_connect, block)
            print(f'第{i}个轨迹：',move_positions[i])
            if moves_result != 0:
                print(f"\nmoves操作失败，错误码: {moves_result}, 在位置: {pos}\n")
                return

        print("\nmoves操作成功\n")

    def demo_rm_get_current_arm_state(self):
        result = self.robot.rm_get_current_arm_state()
        all_pose = result[1]
        all_pose = all_pose['pose']
        all_quat = all_pose[-3:]   # 保存采集的rpy
        all_pose = all_pose[:3]    # 保存采集的坐标
        return all_pose, all_quat

    def stop(self):
        self.robot.rm_set_arm_slow_stop()
        self.robot.rm_set_delete_current_trajectory()


def main(robot_controller, custom_positions=None, block=1):
    # 获取API版本
    print("\nAPI版本: ", rm_api_version(), "\n")

    # 执行moves运动
    # robot_controller.moves()

    # trajectory_array = np.load('further_downsampled_close_trajectory_with_rpy.npy')

    #print("trajectory:\n", trajectory_array)

    # 定义自定义移动位置
    # custom_positions = trajectory_array

    # # 执行移动操作
    robot_controller.moves(custom_positions)


if __name__ == "__main__":
    # 创建机械臂控制器实例并连接到机械臂
    robot_controller = RobotArmController("192.168.33.80", 8080, 3)

    main(robot_controller)

    # 断开与机械臂的连接
    robot_controller.disconnect()
