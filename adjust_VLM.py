# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from Robotic.rm_robot_interface import *
from PIL import Image, ImageDraw
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


def rpy_to_rotation_matrix(rx, ry, rz):
    # 将角度转换为弧度
    #rx, ry, rz = np.radians([rx, ry, rz])
    
    # 计算绕X轴、Y轴和Z轴的旋转矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵，得到从基座坐标系到末端坐标系的旋转矩阵
    R_base = np.dot(R_z, np.dot(R_y, R_x))
    
    return R_base

# 获取对齐图像帧与相机参数
def get_aligned_images():
    # 设置RealSense
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流

    pipe_profile = pipeline.start(config)  # streaming流开始

    # 创建对齐对象与color流对齐
    align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐
    frames = pipeline.wait_for_frames()     # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)      # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()      # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()      # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参

    #### 将images转为numpy arrays ####
    img_color = np.asanyarray(aligned_color_frame.get_data())       # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())       # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame

# 获取随机点三维坐标
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    
    # 确保坐标在图像范围内
    if x < 0 or x >= aligned_depth_frame.width or y < 0 or y >= aligned_depth_frame.height:
        raise ValueError(f"Depth pixel coordinate ({x}, {y}) is out of bounds.")
    
    # 获取深度值
    dis = aligned_depth_frame.get_distance(x, y)
    
    # 这里要看情况改 太近了会无法检测到
    if dis < 0.1:
        dis = 0.1
        print(f"Invalid depth value at ({x}, {y}): {dis} meters.")
    
    # 将像素坐标转换为相机坐标
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    
    return dis, camera_coordinate


def main(object='', arm=None):
    # 加载CLIPSeg模型和处理器
    model_path = "./models/clipseg-rd64-refined/"
    model = CLIPSegForImageSegmentation.from_pretrained(model_path)
    processor = CLIPSegProcessor.from_pretrained(model_path)

    # 定义图像路径和文本提示
    text_prompt = object

    # 获取对齐图像帧与相机参数
    color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数

    # 将图像保存为初始照片F
    initial_image_path = "initial_image.jpg"
    cv2.imwrite(initial_image_path, img_color)

    # 加载初始照片
    image = Image.open(initial_image_path)

    # 处理图像和文本
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")

    # 计算分割掩码
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.sigmoid()  # 获取分割掩码

    # 将分割掩码转换为二值图像
    mask = preds[0].cpu().numpy() > 0.5
    
    # 将掩膜转换为OpenCV格式
    mask_image = Image.fromarray(np.uint8(mask * 255)).resize(image.size, Image.NEAREST)
    mask_image = np.array(mask_image) > 0
    
    # 应用形态学操作（去噪、平滑掩膜）
    kernel = np.ones((5, 5), np.uint8)  # 创建5x5的结构元素
    mask_image = cv2.morphologyEx(mask_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)  # 闭操作，填补小空洞
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开操作，去除小噪声

        # 初始化新的掩膜
    new_mask = np.zeros_like(mask_image)

    # 在掩膜上绘制小圆
    M = cv2.moments(mask_image)  # 计算掩膜的矩
    if M["m00"] != 0:  # 如果掩膜区域不为空
        # 计算质心
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 在质心处绘制一个小圆
        radius = 70  # 半径可以调整
        cv2.circle(new_mask, (cx, cy), radius, 255, -1)  # 在质心位置绘制白色圆形
    else:
        print("掩膜为空，无法计算质心并绘制圆形")
        
    # 创建一个新的图像，用于显示结果
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # 等待按键输入
    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('a'):  # 检测 'a' 键
    #         break
    
    # 确保掩膜尺寸与原图一致
    new_mask = Image.fromarray(np.uint8(new_mask * 255)).resize(image.size, Image.NEAREST)
    new_mask = np.array(new_mask) > 0
    
    # 创建一个新的图像，用于显示结果
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    # 获取分割掩码的边界框
    mask_indices = np.where(new_mask)
    
    if mask_indices[0].size > 0 and mask_indices[1].size > 0:
        ymin, ymax = mask_indices[0].min(), mask_indices[0].max()
        xmin, xmax = mask_indices[1].min(), mask_indices[1].max()

        # 在原图上绘制框
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), text_prompt, fill="red")

        # 保存结果图像
        result_image_path = text_prompt + ".jpg"
        result_image.save(result_image_path)

        # 读取已保存的图片
        saved_img = cv2.imread(result_image_path)

        # 重新检测红色框选区域
        x_saved, y_saved, w_saved, h_saved, center_x_saved, center_y_saved = xmin, ymin, xmax-xmin, ymax-ymin, (xmin+xmax)//2, (ymin+ymax)//2

        # 确保检测到的坐标在深度图像的范围内
        if 0 <= center_x_saved < aligned_depth_frame.width and 0 <= center_y_saved < aligned_depth_frame.height:
            # 获取中心点的三维坐标
            depth_pixel_saved = [center_x_saved, center_y_saved]
            dis_saved, camera_coordinate_saved = get_3d_camera_coordinate(depth_pixel_saved, aligned_depth_frame, depth_intrin)

            # 转换为以向右为x, 向后为y, 向下为z的坐标系，与D435i坐标系相同
            x_3d_saved = camera_coordinate_saved[0]
            y_3d_saved = camera_coordinate_saved[1]
            z_3d_saved = camera_coordinate_saved[2]
            
            # print("Success Detect")
            
            # 定义原始坐标
            point = np.array([[x_3d_saved], [y_3d_saved], [z_3d_saved]])
            
            # print("x坐标:",x_3d_saved)
            # print("y坐标:",y_3d_saved)
            # print("z坐标:",z_3d_saved)
            
            # 标记框选区域和中心点
            cv2.circle(saved_img, (center_x_saved, center_y_saved), 8, [255, 0, 255], thickness=-1)
            cv2.putText(saved_img, "X:" + str(x_3d_saved) + " m", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
            cv2.putText(saved_img, "Y:" + str(y_3d_saved) + " m", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
            cv2.putText(saved_img, "Z:" + str(z_3d_saved) + " m", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 0])
            
            
            # 定义相机与末端坐标系的旋转矩阵
            rotation_matrix = np.array([
            [ 0.02193009, 0.99956877, -0.01952805],
            [-0.99975102, 0.02184521, -0.00454894],
            [-0.00412039, 0.01962295, 0.99979896]
            ])
            
            # 定义相机与末端坐标系的平移向量
            translation_vector = np.array([[-0.04575812], [0.03090394], [0.01780443]])
            
            # 计算旋转后的坐标
            rotated_point = np.dot(rotation_matrix, point)

            # 计算旋转和平移后的新坐标
            point_camera_to_mo = rotated_point + translation_vector
            
            current_state = arm.demo_rm_get_current_arm_state()
            
            # 末端位姿
            pose_x_mo = current_state[0][0]
            pose_y_mo = current_state[0][1]
            pose_z_mo = current_state[0][2]

            roll_rad_mo = current_state[1][0]
            pitch_rad_mo = current_state[1][1]
            yaw_rad_mo = current_state[1][2]

            # 末端坐标
            pose_mo = np.array([pose_x_mo ,pose_y_mo ,pose_z_mo])

            # 末端坐标系与基坐标系的旋转矩阵
            rotated_mo_to_basic = rpy_to_rotation_matrix(roll_rad_mo,pitch_rad_mo,yaw_rad_mo)

            rotated_point_mo_to_basic = np.dot(rotated_mo_to_basic , point_camera_to_mo)

            # 平移向量
            translation_mo_to_basic = np.array([[pose_x_mo], [pose_y_mo], [pose_z_mo]])

            # 计算旋转和平移后的新坐标
            point_mo_to_basic = rotated_point_mo_to_basic + translation_mo_to_basic
            
            # 定义勺子平移向量
            translation_vector_shaozi = np.array([[0.03], [0], [0]])
            
            return_point = point_mo_to_basic + translation_vector_shaozi
            
            print("原始坐标：\n", point)
            print("最终坐标：\n", return_point)
        else:
            print(f"Detected red box center ({center_x_saved}, {center_y_saved}) is out of depth frame bounds.")

    else:
        print("no detect")

    # 释放资源
    # pipeline.stop()
    cv2.destroyAllWindows()
    flattened = [item[0] for item in return_point]
    flattened += [roll_rad_mo, pitch_rad_mo, yaw_rad_mo]
    return flattened


if __name__ == "__main__":
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建机械臂连接，打印连接id
    handle = arm.rm_create_robot_arm("192.168.33.80", 8080)

    main('sugar', handle)
