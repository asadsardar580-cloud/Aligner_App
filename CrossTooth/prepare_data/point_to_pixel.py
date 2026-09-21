import trimesh
import pyrender
import numpy as np
import os
import math
import matplotlib.pyplot as plt


def m3dLookAt(eye, target, up):
    def normalize(v):
        return v / np.sqrt(np.sum(v ** 2))

    mz = normalize(eye - target)
    mx = normalize(np.cross(up, mz))
    my = normalize(np.cross(mz, mx))

    return np.array([
        [mx[0], my[0], mz[0], eye[0]],
        [mx[1], my[1], mz[1], eye[1]],
        [mx[2], my[2], mz[2], eye[2]],
        [0, 0, 0, 1]
    ])


def project_point(rt, k, point):
    # 将点P的三维坐标转换为齐次坐标形式
    point_homogeneous = np.array([point[0], point[1], point[2], 1])

    # 将点P的三维坐标转换到相机坐标系中
    point_cam = np.dot(rt, point_homogeneous)
    point_cam = point_cam[:3]  # 忽略最后一个值1

    # 将点P在相机坐标系中的坐标转换到图像平面坐标系中
    point_img = np.dot(k, point_cam)

    # 将图像平面坐标系中的坐标转换为二维坐标形式
    img_x = point_img[0] / point_img[2]
    img_y = point_img[1] / point_img[2]

    return img_x, img_y


def point_to_pixel(model_path, img_path, point, rend_size=(256, 256)):
    print(model_path)
    base_name = os.path.basename(model_path)[:-4]

    fuze_trimesh = trimesh.load(model_path)
    vertices = np.asarray(fuze_trimesh.vertices)
    minCoord = np.min(vertices, axis=0)
    maxCoord = np.max(vertices, axis=0)
    meanCoord = np.mean(vertices, axis=0)
    x_len, y_len, z_len = maxCoord - minCoord
    # 计算球体半径
    radius = np.sqrt(np.sum((maxCoord - meanCoord) ** 2)) * 1.2
    theta = math.asin((z_len / 2) / radius)

    # 记录所有的相机位姿
    camera_pos_list = []

    while theta <= math.pi / 2:
        z_offset = math.sin(theta) * radius
        sub_radius = math.cos(theta) * radius
        beta = 0
        while beta <= math.pi * 2:
            x_offset = sub_radius * math.cos(beta)
            y_offset = sub_radius * math.sin(beta)
            camera_pos = meanCoord + np.asarray([x_offset, y_offset, z_offset], dtype=float)

            # 计算一个上方向
            camera_pos_list.append((
                m3dLookAt(camera_pos, meanCoord, np.asarray([0, 0, 1], dtype=float)),
                theta,
                beta,
                camera_pos
            ))
            beta += math.pi / 6
        theta += math.pi / 9

    # 创建模型
    pyrender_mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

    # 创建场景
    scene = pyrender.Scene()

    # 场景添加模型
    scene.add(pyrender_mesh)

    # 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(light)

    # 创建相机
    yfov = np.pi / 2
    aspectRatio = 1.0

    # 相机的外参(Rt)是指相机的旋转矩阵(R)和平移向量(t)的组合。它描述了相机在世界坐标系中的位置和朝向。
    # 相机的内参(K)是指相机的内部参数矩阵。它包含了相机的焦距、光心位置和像素尺寸等信息。内参矩阵通常表示为一个3x3的矩阵，包括焦距(fx, fy)、光心位置(cx, cy)和像素尺寸(sx, sy)等参数。
    camera_rt = camera_pos_list[0][0]
    Rt = np.eye(4)
    Rt[:3, :3] = camera_rt[:3, :3].T
    Rt[:3, 3] = -np.dot(camera_rt[:3, :3].T, camera_rt[:3, 3])
    # 焦距f
    f_y = (rend_size[0] / 2) / math.tan(yfov / 2)
    f_x = f_y * aspectRatio
    # 光心c
    cx, cy = rend_size[1] / 2.0, rend_size[0] / 2.0
    K = np.array(
        [[f_x, 0, cx],
         [0, f_y, cy],
         [0, 0, 1]]
    )
    camera = pyrender.IntrinsicsCamera(fx=f_x, fy=f_y, cx=cx, cy=cy)

    # 渲染参数
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)

    # 渲染图片
    camera_node = scene.add(camera, pose=camera_pos_list[0][0])

    # 找到点云对应的像素坐标
    pixel_point = project_point(Rt, K, point)
    pixel_point = [pixel_point[1], rend_size[0] - pixel_point[0]]
    pixel_point = [round(pixel_point[0]), round(pixel_point[1])]

    # 输出像素坐标
    print(pixel_point)

    color, depth = r.render(scene)

    # 设置该点的颜色
    color = color.copy()
    for x in range(max(pixel_point[0] - 3, 0), min(pixel_point[0] + 3, rend_size[0] - 1)):
        for y in range(max(pixel_point[1] - 3, 0), min(pixel_point[1] + 3, rend_size[1] - 1)):
            color[x, y] = np.asarray([255, 0, 0])

    plt.imsave(img_path, color)

    r.delete()


if __name__ == '__main__':
    visible_point = np.asarray([-0.756694, -12.8739, -86.211])
    invisible_point = np.asarray([1.35, -11.35, -89.15])
    point_to_pixel(
        'ENQ0R553_lower.obj',
        'visible_point.png',
        point=visible_point,
        rend_size=(1024, 1024)
    )
