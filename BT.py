import sys
import select
import main_xyz
import main_rpy
import threading
import time
from typing import Optional
import numpy as np
import adjust_VLM as aVLM
import finally_VLM as fVLM
import check_sugar as ck
from RMDemo_Moves.src.core import test
from RMDemo_Moves.src.core import demo_moves as move

# 全局状态和通信工具
# class SharedState:
#     def __init__(self):
#         self.stop_event = threading.Event()  # 终止信号
#         self.abnormal_reason: Optional[str] = None  # 存储异常原因
#
# # ------------------------- 监视函数逻辑 -------------------------
# def monitor_function(shared_state: SharedState):
#     """持续监视实验场景，检测到异常时触发终止信号"""
#     try:
#         while not shared_state.stop_event.is_set():
#             # 非阻塞读取输入
#             if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
#                 key = sys.stdin.read(1)
#                 if key.lower() == 'a':
#                     print("\n检测到停止指令！")
#                     shared_state.abnormal_reason = "手动终止"
#                     shared_state.stop_event.set()
#                     robot_controller.stop()  # 假设这是你的停止方法
#                     break
#
#     except Exception as e:
#         print(f"监视函数发生错误: {e}")
#         shared_state.stop_event.set()

def move_target(robot, task, xyz=None, rpy=None, error_range=0.1):
    if task == "home":
        move.main(robot)
    else:
        # current_state = robot.demo_rm_get_current_arm_state()
        # main_xyz.main(current_state[0], xyz, task, error_range)
        # main_rpy.main(current_state[1], rpy, task)
        # test.main(task)
        move.main(robot, np.load(f'RMDemo_Moves/src/core/further_downsampled_{task}_trajectory_with_rpy.npy'))

def rotate(robot, task, num=49):
    current_state = robot.demo_rm_get_current_arm_state()
    current_state_xyz = current_state[0]
    if task == 'take':
        move_positions =[
            [-0.35928314, -0.082496, 0.16535923, 3.10053133, 0.05633168, 0.23558434],
            [-0.35928314, -0.082496, 0.16535923, 3.10390109, 0.11520733, 0.2328796],
            [-0.35928314, -0.082496, 0.16535923, 3.10673492, 0.16666222, 0.2306634],
            [-0.35928314, -0.082496, 0.16535923, 3.10909816, 0.21082945, 0.22886162],
            [-0.35928314, -0.082496, 0.16535923, 3.11104969, 0.24813753, 0.22740889],
            [-0.35928314, -0.082496, 0.16535923, 3.11272036, 0.28057721, 0.22619308],
            [-0.35928314, -0.082496, 0.16535923, 3.11412124, 0.30785139, 0.22519582],
            # [-0.35928314, -0.082496, 0.16535923, 3.11527002, 0.3296154, 0.22439608],
            # [-0.35928314, -0.082496, 0.16535923, 3.11624744, 0.34678596, 0.22373144],
            # [-0.35928314, -0.082496, 0.16535923, 3.11708612, 0.35992942, 0.223174],
            # [-0.35928314, -0.082496, 0.16535923, 3.11781857, 0.37013066, 0.22269639],
            # [-0.35928314, -0.082496, 0.16535923, 3.11849588, 0.37865869, 0.22226136],
            # [-0.35928314, -0.082496, 0.16535923, 3.11912041, 0.38591847, 0.22186503],
            # [-0.35928314, -0.082496, 0.16535923, 3.11970105, 0.39227109, 0.22150015],
            # [-0.35928314, -0.082496, 0.16535923, 3.12023488, 0.39785247, 0.22116742],
            # [-0.35928314, -0.082496, 0.16535923, 3.12074362, 0.40299204, 0.22085256],
            # [-0.35928314, -0.082496, 0.16535923, 3.1212211, 0.4076901, 0.22055891],
            # [-0.35928314, -0.082496, 0.16535923, 3.12166264, 0.41194774, 0.22028888],
            # [-0.35928314, -0.082496, 0.16535923, 3.12208502, 0.41595707, 0.22003191],
            # [-0.35928314, -0.082496, 0.16535923, 3.12248246, 0.41968294, 0.21979126],
            # [-0.35928314, -0.082496, 0.16535923, 3.12285064, 0.42310068, 0.21956931],
            # [-0.35928314, -0.082496, 0.16535923, 3.1232033, 0.42634846, 0.21935759],
            # [-0.35928314, -0.082496, 0.16535923, 3.12353545, 0.42938769, 0.21915896],
            # [-0.35928314, -0.082496, 0.16535923, 3.12384337, 0.43219041, 0.21897549],
            # [-0.35928314, -0.082496, 0.16535923, 3.12413847, 0.43486474, 0.21880025],
            # [-0.35928314, -0.082496, 0.16535923, 3.12441654, 0.43737554, 0.21863567],
            # [-0.35928314, -0.082496, 0.16535923, 3.12467859, 0.43973439, 0.21848105],
            # [-0.35928314, -0.082496, 0.16535923, 3.12492162, 0.44191647, 0.21833806],
            # [-0.35928314, -0.082496, 0.16535923, 3.12515463, 0.44400381, 0.21820134],
            # [-0.35928314, -0.082496, 0.16535923, 3.12537425, 0.44596752, 0.21807281],
            # [-0.35928314, -0.082496, 0.16535923, 3.12557798, 0.44778607, 0.21795388],
            # [-0.35928314, -0.082496, 0.16535923, 3.12577333, 0.44952727, 0.2178401],
            # [-0.35928314, -0.082496, 0.16535923, 3.12595749, 0.45116659, 0.21773308],
            # [-0.35928314, -0.082496, 0.16535923, 3.12612834, 0.45268572, 0.21763401],
            # [-0.35928314, -0.082496, 0.16535923, 3.12629217, 0.45414101, 0.21753918],
            # [-0.35928314, -0.082496, 0.16535923, 3.12644663, 0.45551179, 0.21744995],
            # [-0.35928314, -0.082496, 0.16535923, 3.12658995, 0.45678256, 0.21736731],
            # [-0.35928314, -0.082496, 0.16535923, 3.12672738, 0.45800033, 0.21728819],
            # [-0.35928314, -0.082496, 0.16535923, 3.12685697, 0.45914772, 0.21721371],
            # [-0.35928314, -0.082496, 0.16535923, 3.12697915, 0.46022888, 0.2171436],
            # [-0.35928314, -0.082496, 0.16535923, 3.12709252, 0.46123147, 0.21707863],
            # [-0.35928314, -0.082496, 0.16535923, 3.12720125, 0.46219252, 0.21701641],
            # [-0.35928314, -0.082496, 0.16535923, 3.12730378, 0.46309824, 0.21695782],
            # [-0.35928314, -0.082496, 0.16535923, 3.12739891, 0.46393826, 0.21690351],
            # [-0.35928314, -0.082496, 0.16535923, 3.12749016, 0.46474358, 0.21685149],
            # [-0.35928314, -0.082496, 0.16535923, 3.1275762, 0.46550263, 0.21680249],
            # [-0.35928314, -0.082496, 0.16535923, 3.12765603, 0.4662067, 0.21675707],
            # [-0.35928314, -0.082496, 0.16535923, 3.12773261, 0.46688174, 0.21671355],
            # [-0.35928314, -0.082496, 0.16535923, 3.12780482, 0.46751804, 0.21667255],
            # [-0.35928314, -0.082496, 0.16535923, 3.12787291, 0.46811785, 0.21663392]
            ]
        move_positions = [
            current_state_xyz + move_position[3:] for move_position in move_positions
        ]
        move.main(robot, move_positions)
    elif task == 'put':
        move_positions = [
            [-0.43735376, 0.11363103, 0.23252343, 2.892, 0.51, -0.271],
            [-0.43735376, 0.11363103, 0.23252343, 2.48980765, 0.20405968, -0.7188785],
            [-0.43735376, 0.11363103, 0.23252343, 2.29245218, -0.11891064, -0.96281573],
            [-0.43735376, 0.11363103, 0.23252343, 2.18763629, -0.39634814, -1.11867852],
            [-0.43735376, 0.11363103, 0.23252343, 2.13287803, -0.61252842, -1.22864644],
            [-0.43735376, 0.11363103, 0.23252343, 2.11287356, -0.7290196, -1.28845431],
            [-0.43735376, 0.11363103, 0.23252343, 2.10415533, -0.80373475, -1.32846422],
            [-0.43735376, 0.11363103, 0.23252343, 2.09982205, -0.86675105, -1.36399656],
            [-0.43735376, 0.11363103, 0.23252343, 2.09878173, -0.92199715, -1.3971174],
            [-0.43735376, 0.11363103, 0.23252343, 2.10054221, -0.9716216, -1.42902878],
            [-0.43735376, 0.11363103, 0.23252343, 2.1048293, -1.01634674, -1.46012624],
            [-0.43735376, 0.11363103, 0.23252343, 2.11142158, -1.05668023, -1.49065429],
            [-0.43735376, 0.11363103, 0.23252343, 2.12012165, -1.09304281, -1.52078029],
            [-0.43735376, 0.11363103, 0.23252343, 2.130744, -1.12580545, -1.55061918],
            [-0.43735376, 0.11363103, 0.23252343, 2.14324113, -1.15558506, -1.58054255],
            [-0.43735376, 0.11363103, 0.23252343, 2.15718146, -1.18209267, -1.60998974],
            [-0.43735376, 0.11363103, 0.23252343, 2.17249923, -1.20591927, -1.63926719],
            [-0.43735376, 0.11363103, 0.23252343, 2.18900887, -1.22731804, -1.66835384],
            [-0.43735376, 0.11363103, 0.23252343, 2.20652152, -1.24652033, -1.69720536],
            [-0.43735376, 0.11363103, 0.23252343, 2.22484602, -1.26373754, -1.72575814],
            [-0.43735376, 0.11363103, 0.23252343, 2.24379087, -1.27916274, -1.7539337],
            [-0.43735376, 0.11363103, 0.23252343, 2.26336401, -1.29310404, -1.78192042],
            [-0.43735376, 0.11363103, 0.23252343, 2.28298822, -1.30544472, -1.80906309],
            [-0.43735376, 0.11363103, 0.23252343, 2.30268138, -1.31647848, -1.83554809],
            [-0.43735376, 0.11363103, 0.23252343, 2.32227862, -1.32633856, -1.86128215],
            [-0.43735376, 0.11363103, 0.23252343, 2.34162881, -1.33514602, -1.88617839],
            [-0.43735376, 0.11363103, 0.23252343, 2.36059669, -1.34301064, -1.91015921],
            [-0.43735376, 0.11363103, 0.23252343, 2.37906434, -1.35003178, -1.93315856],
            [-0.43735376, 0.11363103, 0.23252343, 2.39710918, -1.35635889, -1.95533986],
            [-0.43735376, 0.11363103, 0.23252343, 2.4142884, -1.36194692, -1.97621992],
            [-0.43735376, 0.11363103, 0.23252343, 2.43072244, -1.36693537, -1.99600054],
            [-0.43735376, 0.11363103, 0.23252343, 2.4463652, -1.37138938, -2.01466972],
            [-0.43735376, 0.11363103, 0.23252343, 2.4611863, -1.37536724, -2.03222789],
            [-0.43735376, 0.11363103, 0.23252343, 2.47516954, -1.37892104, -2.04868665],
            [-0.43735376, 0.11363103, 0.23252343, 2.48831114, -1.38209725, -2.06406731],
            [-0.43735376, 0.11363103, 0.23252343, 2.50073807, -1.38496437, -2.07853895],
            [-0.43735376, 0.11363103, 0.23252343, 2.51221768, -1.38750218, -2.09184866],
            [-0.43735376, 0.11363103, 0.23252343, 2.52290138, -1.38977369, -2.10418782],
            [-0.43735376, 0.11363103, 0.23252343, 2.53281769, -1.39180796, -2.11560166],
            [-0.43735376, 0.11363103, 0.23252343, 2.54199933, -1.39363076, -2.12613799],
            [-0.43735376, 0.11363103, 0.23252343, 2.55048193, -1.39526501, -2.1358461],
            [-0.43735376, 0.11363103, 0.23252343, 2.55830304, -1.39673102, -2.14477585],
            [-0.43735376, 0.11363103, 0.23252343, 2.5655708, -1.39805943, -2.15305621],
            [-0.43735376, 0.11363103, 0.23252343, 2.57217896, -1.3992398, -2.16057081],
            [-0.43735376, 0.11363103, 0.23252343, 2.57824166, -1.40030037, -2.16745355],
            [-0.43735376, 0.11363103, 0.23252343, 2.58379646, -1.40125378, -2.17375024],
            [-0.43735376, 0.11363103, 0.23252343, 2.58887975, -1.40211125, -2.17950472],
            [-0.43735376, 0.11363103, 0.23252343, 2.59352645, -1.40288281, -2.18475867],
            [-0.43735376, 0.11363103, 0.23252343, 2.59776987, -1.40357734, -2.18955146],
            [-0.43735376, 0.11363103, 0.23252343, 2.60167885, -1.40420879, -2.19396225],
                ]
        move_positions = move_positions[:num]
        current_state_xyz[1] += 0.03
        move_positions = [
            current_state_xyz + move_position[3:] for move_position in move_positions
        ]
        move.main(robot, move_positions)
    elif task == 'rise_take':
        main_rpy.main(current_state[1], [3.127,0.771,-0.083], task)
        move_rpys = np.load('data/rise_take_rpy_results.npy')[:60]
        print(current_state_xyz)
        print(move_rpys)
        move_positions = [
            current_state_xyz + move_rpy.tolist() for move_rpy in move_rpys
        ]
        move.main(robot, move_positions)
    else:
        print(f"任务选择错误，无法执行{task}")

x_s, y_s, z_s, x_o, y_o, z_o = 0, 0, 0, 0, 0, 0
key = 'b'
# 创建机械臂控制器实例并连接到机械臂
robot_controller = move.RobotArmController("192.168.33.80", 8080, 3)

def wait(num):
    global key
    while True:
        time.sleep(0.5)
        print(num)
        # 非阻塞读取输入
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input = sys.stdin.read(1)
            if input.lower() == 'r':
                print("\n检测到停止指令！")
                robot_controller.stop()  # 假设这是你的停止方法
                key = 'r'
                break
            elif input.lower() == 'b':
                key = 'b'
                break

def func0():
    print("回家")
    move_target(robot_controller, task="home")
    wait(0)

def func1():
    global x_s, y_s, z_s
    x_s, y_s, z_s = fVLM.main('sugar', [[0.062], [0.0], [0.14]])

def func2():
    global x_o, y_o, z_o
    x_o, y_o, z_o = fVLM.main('coffee', [[0.062], [0.0], [0.14]])

def func3():
    print("找糖")
    global x_s, y_s, z_s
    move_target(robot_controller, 'circle_close',  [x_s, y_s, z_s], [3.14, 0.209, -0.084], 0.04)
    wait(1)
    if key == 'b':
        x_s, y_s, _, s_r, s_p, s_y = aVLM.main('sugar', robot_controller)
        move.main(robot_controller, [[x_s + 0.02, y_s - 0.01, z_s - 0.07, s_r, s_p, s_y]])
        wait(1.1)
    elif key == 'r':
        return

def func4():
    print("勺糖")
    rotate(robot_controller, 'rise_take')
    current_state = robot_controller.demo_rm_get_current_arm_state()
    move.main(robot_controller, [[x_s, y_s, z_s] + current_state[1]])
    wait(2)

def func5():
    print("找咖啡")
    global x_o, y_o, z_o
    move_target(robot_controller, 'circle_lift', [x_o + 0.1, y_o, 0.21], [3.13447957, 0.46433679, -0.08256184], 0.05)
    wait(3)
    x_o, y_o, _, o_r, o_p, o_y = aVLM.main('coffee', robot_controller)
    move.main(robot_controller, [[x_o, y_o - 0.03, 0.21, o_r, o_p, o_y]])
    wait(3.1)

def func6():
    print("倒糖")
    rotate(robot_controller, 'put', 10)
    wait(4)

dic1 = {
    4: func4,
    5: func5,
    6: func6
}

dict2 = {
    1: func1,
    3: func3,
    4: func4,
    5: func5,
    6: func6
}

def main():
    func1()
    # func2()
    #
    # # move_target(robot_controller, [x_s, y_s, z_s], [3.097,0.047,0.234], 'close', 0.03)
    # # rotate(robot_controller, 'take')
    # # move_target(robot_controller, [x_o, y_o, z_o], [3.117,0.296,0.015], 'lift', 0.04)
    #
    func3()
    if key == 'b':
        func4()
        func5()
        func6()
    elif key == 'r':
        func0()
        func1()
        func3()
        func4()
        func5()
        func6()
    #
    # func4(robot_controller)
    # # move.main(robot_controller, [[x_s+0.02, y_s, z_s-0.065, s_r, s_p, s_y]])
    # # rotate(robot_controller, 'rise_take')
    # # move.main(robot_controller, [[x_s, y_s, z_s] + current_state[1]])
    #
    # # import time
    # # time.sleep(3)
    # # move.main(robot_controller, [[x_s+0.02, y_s, z_s-0.065, s_r, s_p, s_y]])
    # # rotate(robot_controller, 'rise_take')
    # # move.main(robot_controller, [[x_s, y_s, z_s] + current_state[1]])
    #
    # func5(robot_controller)
    #
    # func6(robot_controller)
    #
    # # import time
    # # time.sleep(3)
    # # move.main(robot_controller, [[x_o, y_o, 0.21, o_r, o_p, o_y]])
    # # rotate(robot_controller, 'put')


    move_positions = [
        [-0.205286,0.016923,0.419555,3.142,0.209,-0.084]
    ]
    move.main(robot_controller, move_positions)

    # 断开与机械臂的连接
    robot_controller.disconnect()

if __name__ == '__main__':
    # shared_state = SharedState()
    # # 创建并启动线程
    # main_thread = threading.Thread(target=main, args=(shared_state,))
    # monitor_thread = threading.Thread(target=monitor_function, args=(shared_state,))
    # main_thread.start()
    # monitor_thread.start()
    #
    # # 等待主线程结束（监视线程会随主线程退出）
    # main_thread.join()
    # monitor_thread.join()

    main()
