import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
from point_to_pixel import project_point
from utils import color2label, label2color_lower


def show_ply(vertices, colors, save_path):
    header = (f"ply\n"
              f"format ascii 1.0\n"
              f"comment VCGLIB generated\n"
              f"element vertex {vertices.shape[0]}\n"
              f"property double x\n"
              f"property double y\n"
              f"property double z\n"
              f"property uchar red\n"
              f"property uchar green\n"
              f"property uchar blue\n"
              f"property uchar alpha\n"
              f"element face {0}\n"
              f"property list uchar int vertex_indices\n"
              f"property uchar red\n"
              f"property uchar green\n"
              f"property uchar blue\n"
              f"property uchar alpha\n"
              f"end_header\n")

    vertex_info = ""
    for color, coord in zip(colors, vertices):
        vertex_info += f'{coord[0]} {coord[1]} {coord[2]} {color[0]} {color[1]} {color[2]} {255}\n'

    with open(save_path, 'w', encoding='ascii') as f:
        f.write(header)
        f.write(vertex_info)


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


# 判断点是否在三角形内
def is_point_in_triangle(px, py, x1, y1, x2, y2, x3, y3):
    def area(x1, y1, x2, y2, x3, y3):
        return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    total_area = area(x1, y1, x2, y2, x3, y3)
    area1 = area(px, py, x2, y2, x3, y3)
    area2 = area(x1, y1, px, py, x3, y3)
    area3 = area(x1, y1, x2, y2, px, py)

    return abs(total_area - (area1 + area2 + area3)) < 1e-3


# 获取三角形覆盖的坐标
def get_covered_points(x1, y1, x2, y2, x3, y3):
    min_x = min(x1, x2, x3)
    max_x = max(x1, x2, x3)
    min_y = min(y1, y2, y3)
    max_y = max(y1, y2, y3)

    covered_points = []

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if is_point_in_triangle(x, y, x1, y1, x2, y2, x3, y3):
                covered_points.append((x, y))

    return covered_points


def render(model_path, label_path, save_path, rend_size=(256, 256)):
    print(model_path)
    base_name = os.path.basename(model_path)[:-4]

    os.makedirs(os.path.join(save_path, base_name), exist_ok=True)
    os.makedirs(os.path.join(save_path, base_name, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, base_name, 'label'), exist_ok=True)
    os.makedirs(os.path.join(save_path, base_name, 'p2p'), exist_ok=True)

    fuze_trimesh = trimesh.load(model_path)
    vertices = np.asarray(fuze_trimesh.vertices)
    num_cells = len(fuze_trimesh.faces)
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
    label_trimesh = trimesh.load(label_path)

    # 创建场景
    scene = pyrender.Scene()
    label_scene = pyrender.Scene()

    # 场景添加模型
    scene.add(pyrender_mesh)
    # 标签实例分割需要获取单独实例
    seg_node_map = {}
    vertex_instances = {}
    label_color_map = {}
    face_instances = {}
    for i, vertex_color in enumerate(label_trimesh.visual.vertex_colors):
        vertex_color = (vertex_color[0], vertex_color[1], vertex_color[2])
        if vertex_color in color2label:
            vertex_label = color2label[vertex_color][2]
            if not vertex_label in vertex_instances:
                vertex_instances[vertex_label] = {}
            vertex_instances[vertex_label][i] = len(vertex_instances[vertex_label])
            label_color_map[vertex_label] = vertex_color
    for i, face in enumerate(label_trimesh.faces):
        for label, vertices in vertex_instances.items():
            if face[0] in vertices and face[1] in vertices and face[2] in vertices:
                if not label in face_instances:
                    face_instances[label] = []
                face_instances[label].append([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
    for label, vertices in vertex_instances.items():
        label_color = label_color_map[label]
        label_color = [label_color[0], label_color[1], label_color[2]]
        vertice_node = np.array([label_trimesh.vertices[i] for i, _ in vertices.items()], dtype=float)
        vertice_color_node = np.array([label_color] * vertice_node.shape[0])
        face_node = np.array(face_instances[label])
        face_color_node = np.array([label_color] * face_node.shape[0])
        mesh_node = trimesh.Trimesh(vertices=vertice_node, faces=face_node, vertex_colors=vertice_color_node, face_colors=face_color_node)
        # 当前模型添加到场景中
        node = label_scene.add(pyrender.Mesh.from_trimesh(mesh_node))
        seg_node_map[node] = label_color

    # 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(light)
    label_scene.add(light)

    # 创建相机
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2, aspectRatio=1.0)

    # 渲染参数
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)

    angle_info = ""

    # 每个点的标签投票
    cell_label_vote = np.zeros((num_cells, 17))

    for i, camera_pos in enumerate(camera_pos_list):
        angle_info += f'theta {camera_pos[1]} beta {camera_pos[2]}\n'

        # 渲染图片
        camera_node = scene.add(camera, pose=camera_pos[0])
        img_color, img_depth = r.render(scene)
        scene.remove_node(camera_node)
        plt.imsave(os.path.join(save_path, base_name, 'img', f'{base_name}_{i}.png'), img_color)

        # 渲染分割标签
        camera_node = label_scene.add(camera, pose=camera_pos[0])
        label_color, label_depth = r.render(label_scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_node_map)
        label_scene.remove_node(camera_node)
        plt.imsave(os.path.join(save_path, base_name, 'label', f'{base_name}_{i}.png'), label_color)

        # 口扫点云坐标对应到图片像素坐标
        point_coords = np.asarray(fuze_trimesh.vertices)
        cells = np.asarray(fuze_trimesh.faces)
        # 面片重心点坐标
        cell_coords = np.array([[
            (point_coords[point_idxs[0]][0] + point_coords[point_idxs[1]][0] + point_coords[point_idxs[2]][0]) / 3,
            (point_coords[point_idxs[0]][1] + point_coords[point_idxs[1]][1] + point_coords[point_idxs[2]][1]) / 3,
            (point_coords[point_idxs[0]][2] + point_coords[point_idxs[1]][2] + point_coords[point_idxs[2]][2]) / 3,
        ] for point_idxs in cells])
        # 面片到相机的距离
        cell_distances = np.sqrt(np.sum((cell_coords - camera_pos[3]) ** 2, axis=1))
        # 从近到远面片索引
        sorted_cell_indices = np.argsort(cell_distances)

        pixel_point_map = np.asarray([[[-1, 0]] * rend_size[0]] * rend_size[1])

        for cell_idx in sorted_cell_indices:
            # 面片到相机距离
            distance = cell_distances[cell_idx]

            # 计算点云坐标到图像坐标
            Rt = np.eye(4)
            Rt[:3, :3] = camera_pos[0][:3, :3].T
            Rt[:3, 3] = -np.dot(camera_pos[0][:3, :3].T, camera_pos[0][:3, 3])
            # 焦距f
            f_y = (rend_size[0] / 2) / math.tan(np.pi / 2 / 2)
            f_x = f_y * 1.0
            # 光心c
            cx, cy = rend_size[1] / 2.0, rend_size[0] / 2.0
            K = np.array(
                [[f_x, 0, cx],
                 [0, f_y, cy],
                 [0, 0, 1]]
            )

            # 面片中点在图像上的坐标
            cell_pixel_point = project_point(Rt, K, cell_coords[cell_idx])
            cell_pixel_point = [cell_pixel_point[1], rend_size[0] - cell_pixel_point[0]]
            cell_pixel_point = [round(cell_pixel_point[0]), round(cell_pixel_point[1])]

            # 超出像素坐标，跳过
            if cell_pixel_point[0] < 0 or cell_pixel_point[0] >= rend_size[0] or cell_pixel_point[1] < 0 or cell_pixel_point[1] >= rend_size[1]:
                continue

            pixel_point_0 = project_point(Rt, K, point_coords[cells[cell_idx][0]])
            pixel_point_0 = [pixel_point_0[1], rend_size[0] - pixel_point_0[0]]
            pixel_point_0 = [round(pixel_point_0[0]), round(pixel_point_0[1])]

            # 超出像素坐标，跳过
            if pixel_point_0[0] < 0 or pixel_point_0[0] >= rend_size[0] or pixel_point_0[1] < 0 or pixel_point_0[1] >= rend_size[1]:
                continue

            pixel_point_1 = project_point(Rt, K, point_coords[cells[cell_idx][1]])
            pixel_point_1 = [pixel_point_1[1], rend_size[0] - pixel_point_1[0]]
            pixel_point_1 = [round(pixel_point_1[0]), round(pixel_point_1[1])]

            # 超出像素坐标，跳过
            if pixel_point_1[0] < 0 or pixel_point_1[0] >= rend_size[0] or pixel_point_1[1] < 0 or pixel_point_1[1] >= rend_size[1]:
                continue

            pixel_point_2 = project_point(Rt, K, point_coords[cells[cell_idx][2]])
            pixel_point_2 = [pixel_point_2[1], rend_size[0] - pixel_point_2[0]]
            pixel_point_2 = [round(pixel_point_2[0]), round(pixel_point_2[1])]

            # 超出像素坐标，跳过
            if pixel_point_2[0] < 0 or pixel_point_2[0] >= rend_size[0] or pixel_point_2[1] < 0 or pixel_point_2[1] >= rend_size[1]:
                continue

            # 三个顶点组成的面片，占据的离散整数坐标
            covered_coords = get_covered_points(pixel_point_0[0], pixel_point_0[1], pixel_point_1[0], pixel_point_1[1], pixel_point_2[0], pixel_point_2[1])
            covered_coords.append((cell_pixel_point[0], cell_pixel_point[1]))

            for pixel_point in covered_coords:
                if pixel_point_map[pixel_point[0], pixel_point[1]][0] < 0:
                    pixel_point_map[pixel_point[0], pixel_point[1]] = np.asarray([cell_idx, distance])
                elif pixel_point_map[pixel_point[0], pixel_point[1]][1] > distance:
                    pixel_point_map[pixel_point[0], pixel_point[1]] = np.asarray([cell_idx, distance])

        # 真正的对应关系
        color = img_color.copy()
        map_color = np.asarray([[125, 125, 125]] * cell_coords.shape[0])
        for x in range(rend_size[0]):
            for y in range(rend_size[1]):
                if pixel_point_map[x, y][0] > 0:
                    # 找到对应的图像标签颜色
                    point_color = pixel_color = label_color[x, y]
                    # 标签黑色表示不是前景，使用原始渲染图片的颜色
                    if pixel_color[0] == 0 and pixel_color[1] == 0 and pixel_color[2] == 0:
                        pixel_color = img_color[x, y]
                        point_color = np.asarray([255, 255, 255])
                    map_color[pixel_point_map[x, y][0]] = point_color
                    color[x, y] = pixel_color
                    point_color = (point_color[0], point_color[1], point_color[2])
                    cell_label_vote[pixel_point_map[x, y][0], color2label[point_color][2] if point_color in color2label else 0] += 1

        # 可视化点云对应关系
        show_ply(cell_coords, map_color, os.path.join(save_path, base_name, 'p2p', f'{base_name}_{i}.ply'))

        # 可视化图片对应关系
        plt.imsave(os.path.join(save_path, base_name, 'p2p', f'{base_name}_{i}.png'), color)

    # 点云标签选择投票最多的，可视化输出
    cell_label_vote = np.argmax(cell_label_vote, axis=1)
    cell_color_vote = np.asarray([label2color_lower[label][2] if label > 0 else (255, 255, 255) for label in cell_label_vote])
    show_ply(cell_coords, cell_color_vote, os.path.join(save_path, base_name, f'{base_name}_vote.ply'))

    with open(os.path.join(os.path.join(save_path, base_name, f'{base_name}_angle.txt')), 'w',
              encoding='ascii') as f:
        f.write(angle_info)

    r.delete()


if __name__ == '__main__':
    render(
        '0AAQ6BO3_lower.obj',
        '0AAQ6BO3_lower.ply',
        'tmp',
        rend_size=(1024, 1024)
    )
    render(
        '0AAQ6BO3_upper.obj',
        '0AAQ6BO3_upper.ply',
        'tmp',
        rend_size=(1024, 1024)
    )


