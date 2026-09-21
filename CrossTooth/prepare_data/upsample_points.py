# 这个脚本的作用是把预测结果重新映射回原始 mesh
import os

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import vedo
from utils import color2label, label2color_lower

lower_palette = np.array(
    [[125, 125, 125]] +
    [[label2color_lower[label][2][0],
      label2color_lower[label][2][1],
      label2color_lower[label][2][2]]
     for label in range(1, 17)], dtype=np.uint8
)


def show_ply(vertices, vertice_colors, faces, face_colors, save_path):
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
              f"element face {faces.shape[0]}\n"
              f"property list uchar int vertex_indices\n"
              f"property uchar red\n"
              f"property uchar green\n"
              f"property uchar blue\n"
              f"property uchar alpha\n"
              f"end_header\n")

    vertex_info = ""
    for color, coord in zip(vertice_colors, vertices):
        vertex_info += f'{coord[0]} {coord[1]} {coord[2]} {color[0]} {color[1]} {color[2]} {255}\n'

    face_info = ""
    for color, cell in zip(face_colors, faces):
        face_info += f'3 {cell[0]} {cell[1]} {cell[2]} {color[0]} {color[1]} {color[2]} {255}\n'

    with open(save_path, 'w', encoding='ascii') as f:
        f.write(header)
        f.write(vertex_info)
        f.write(face_info)


def upsample_to_origin(upsampling_method, predict_file, origin_file, save_file):
    print(predict_file)
    print(save_file)

    predict_mesh = vedo.load(predict_file)
    predict_point_coords = predict_mesh.points()

    predict_cell_coords = np.array([[
                (predict_point_coords[point_idxs[0]][0] + predict_point_coords[point_idxs[1]][0] + predict_point_coords[point_idxs[2]][0]) / 3,
                (predict_point_coords[point_idxs[0]][1] + predict_point_coords[point_idxs[1]][1] + predict_point_coords[point_idxs[2]][1]) / 3,
                (predict_point_coords[point_idxs[0]][2] + predict_point_coords[point_idxs[1]][2] + predict_point_coords[point_idxs[2]][2]) / 3,
            ] for point_idxs in predict_mesh.cells()])
    predict_cell_labels = [color2label[(color[0], color[1], color[2])][2] if (color[0], color[1], color[2]) in color2label else 0 for color in predict_mesh.cellcolors]

    origin_mesh = vedo.load(origin_file)
    origin_point_coords = origin_mesh.points()
    origin_cell_coords = np.array([[
                (origin_point_coords[point_idxs[0]][0] + origin_point_coords[point_idxs[1]][0] + origin_point_coords[point_idxs[2]][0]) / 3,
                (origin_point_coords[point_idxs[0]][1] + origin_point_coords[point_idxs[1]][1] + origin_point_coords[point_idxs[2]][1]) / 3,
                (origin_point_coords[point_idxs[0]][2] + origin_point_coords[point_idxs[1]][2] + origin_point_coords[point_idxs[2]][2]) / 3,
            ] for point_idxs in origin_mesh.cells()])

    if upsampling_method == 'SVM':
        clf = SVC(kernel='rbf', gamma='auto')
        # train SVM
        clf.fit(predict_cell_coords, np.ravel(predict_cell_labels))
        origin_cell_labels = clf.predict(origin_cell_coords)
        origin_cell_labels = origin_cell_labels.reshape(-1)
        origin_point_labels = clf.predict(origin_point_coords)
        origin_point_labels = origin_point_labels.reshape(-1)
    elif upsampling_method == 'KNN':
        neigh = KNeighborsClassifier(n_neighbors=3)
        # train KNN
        neigh.fit(predict_cell_coords, np.ravel(predict_cell_labels))
        origin_cell_labels = neigh.predict(origin_cell_coords)
        origin_cell_labels = origin_cell_labels.reshape(-1)
        origin_point_labels = neigh.predict(origin_point_coords)
        origin_point_labels = origin_point_labels.reshape(-1)

    # 把预测结果输出到文件
    origin_cell_colors = lower_palette[origin_cell_labels]
    origin_point_colors = lower_palette[origin_point_labels]

    show_ply(origin_point_coords, origin_point_colors, np.asarray(origin_mesh.cells()), origin_cell_colors, save_file)

