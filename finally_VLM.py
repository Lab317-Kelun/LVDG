# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from Robotic.rm_robot_interface import *



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
    dis = aligned_depth_frame.get_distance(x, y)        # 获取该像素点对应的深度
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate


def main(object='', translation_vectors=[[0.062], [0.0], [0.06]]):
    # 加载CLIPSeg模型和处理器
    model_path = "./models/clipseg-rd64-refined/"
    model = CLIPSegForImageSegmentation.from_pretrained(model_path)
    processor = CLIPSegProcessor.from_pretrained(model_path)

    # 定义图像路径和文本提示
    text_prompt = object

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建机械臂连接，打印连接id
    handle = arm.rm_create_robot_arm("192.168.33.80", 8080)

    # 获取对齐图像帧与相机参数
    color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数

    # 将图像保存为初始照片
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
    mask_image = (mask * 255).astype(np.uint8)
    
    # 展示掩膜
    cv2.imshow("Mask", mask_image)
    
    # # 等待按键输入
    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('a'):  # 检测 'a' 键
    #         break
    
    # 确保掩膜尺寸与原图一致
    mask = Image.fromarray(np.uint8(mask * 255)).resize(image.size, Image.NEAREST)
    mask = np.array(mask) > 0
    
    # 创建一个新的图像，用于显示结果
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    # 获取分割掩码的边界框
    mask_indices = np.where(mask)
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
            [ 0.00737153, 0.99949606, -0.03087539],
            [-0.99995853, 0.00753303, 0.00511772],
            [0.00534773, 0.03083638, 0.99951014]
            ])
            
            # 定义相机与末端坐标系的平移向量
            translation_vector = np.array([[-0.04612496], [0.03081559], [0.01802954]])
            
            # 计算旋转后的坐标
            rotated_point = np.dot(rotation_matrix, point)

            # 计算旋转和平移后的新坐标
            point_camera_to_mo = rotated_point + translation_vector
            
            current_state = arm.rm_get_current_arm_state()
            
            # 末端位姿
            pose_x_mo = current_state[1]['pose'][0]
            pose_y_mo = current_state[1]['pose'][1]
            pose_z_mo = current_state[1]['pose'][2]

            roll_rad_mo = current_state[1]['pose'][3]
            pitch_rad_mo = current_state[1]['pose'][4]
            yaw_rad_mo = current_state[1]['pose'][5]

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
            translation_vector_shaozi = np.array(translation_vectors)
            
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
    return flattened


if __name__ == "__main__":
    main('robot arm')
