import torch

color2label = {
    # upper
    (170, 255, 127): ("aaff7f", "UL1", 1),
    (170, 255, 255): ("aaffff", "UL2", 2),
    (255, 255, 0): ("ffff00", "UL3", 3),
    (255, 170, 0): ("ffaa00", "UL4", 4),
    (170, 170, 255): ("aaaaff", "UL5", 5),
    (0, 170, 255): ("00aaff", "UL6", 6),
    (85, 170, 0): ("55aa00", "UL7", 7),
    (204, 204, 15): ("cccc0f", "UL8", 8),

    (255, 85, 255): ("ff55ff", "UR1", 9),
    (255, 85, 127): ("ff557f", "UR2", 10),
    (85, 170, 127): ("55aa7f", "UR3", 11),
    (255, 85, 0): ("ff5500", "UR4", 12),
    (0, 85, 255): ("0055ff", "UR5", 13),
    (170, 0, 0): ("aa0000", "UR6", 14),
    (73, 247, 235): ("49f7eb", "UR7", 15),
    (125, 18, 247): ("7d12f7", "UR8", 16),

    # lower
    (240, 0, 0): ("f00000", "LL1", 1),
    (251, 255, 3): ("fbff03", "LL2", 2),
    (44, 251, 255): ("2cfbff", "LL3", 3),
    (241, 47, 255): ("f12fff", "LL4", 4),
    (125, 255, 155): ("7dff9b", "LL5", 5),
    (26, 125, 255): ("1a7dff", "LL6", 6),
    (255, 234, 157): ("ffea9d", "LL7", 7),
    (204, 126, 126): ("cc7e7e", "LL8", 8),

    (206, 129, 212): ("ce81d4", "LR1", 9),
    (45, 135, 66): ("2d8742", "LR2", 10),
    (185, 207, 45): ("b9cf2d", "LR3", 11),
    (69, 147, 207): ("4593cf", "LR4", 12),
    (207, 72, 104): ("cf4868", "LR5", 13),
    (4, 207, 4): ("04cf04", "LR6", 14),
    (35, 1, 207): ("2301cf", "LR7", 15),
    (82, 204, 169): ("52cca9", "LR8", 16),
}

label2color_lower = {
    1: ("f00000", "LL1", (240, 0, 0)),
    2: ("fbff03", "LL2", (251, 255, 3)),
    3: ("2cfbff", "LL3", (44, 251, 255)),
    4: ("f12fff", "LL4", (241, 47, 255)),
    5: ("7dff9b", "LL5", (125, 255, 155)),
    6: ("1a7dff", "LL6", (26, 125, 255)),
    7: ("ffea9d", "LL7", (255, 234, 157)),
    8: ("cc7e7e", "LL8", (204, 126, 126)),

    9: ("ce81d4", "LR1", (206, 129, 212)),
    10: ("2d8742", "LR2", (45, 135, 66)),
    11: ("b9cf2d", "LR3", (185, 207, 45)),
    12: ("4593cf", "LR4", (69, 147, 207)),
    13: ("cf4868", "LR5", (207, 72, 104)),
    14: ("04cf04", "LR6", (4, 207, 4)),
    15: ("2301cf", "LR7", (35, 1, 207)),
    16: ("52cca9", "LR8", (82, 204, 169)),
}

label2color_upper = {
    1: ("aaff7f", "UL1", (170, 255, 127)),
    2: ("aaffff", "UL2", (170, 255, 255)),
    3: ("ffff00", "UL3", (255, 255, 0)),
    4: ("faa00", "UL4", (255, 170, 0)),
    5: ("aaaaff", "UL5", (170, 170, 255)),
    6: ("00aaff", "UL6", (0, 170, 255)),
    7: ("55aa00", "UL7", (85, 170, 0)),
    8: ("cccc0f", "UL8", (204, 204, 15)),

    9: ("ff55ff", "UR1", (255, 85, 255)),
    10: ("ff557f", "UR2", (255, 85, 127)),
    11: ("55aa7f", "UR3", (85, 170, 127)),
    12: ("ff5500", "UR4", (255, 85, 0)),
    13: ("0055ff", "UR5", (0, 85, 255)),
    14: ("aa0000", "UR6", (170, 0, 0)),
    15: ("49f7eb", "UR7", (73, 247, 235)),
    16: ("7d12f7", "UR8", (125, 18, 247)),
}

FDI2color = {
    # upper
    21: ("aaff7f", "UL1", (170, 255, 127)),
    22: ("aaffff", "UL2", (170, 255, 255)),
    23: ("ffff00", "UL3", (255, 255, 0)),
    24: ("faa00", "UL4", (255, 170, 0)),
    25: ("aaaaff", "UL5", (170, 170, 255)),
    26: ("00aaff", "UL6", (0, 170, 255)),
    27: ("55aa00", "UL7", (85, 170, 0)),
    28: ("cccc0f", "UL8", (204, 204, 15)),

    11: ("ff55ff", "UR1", (255, 85, 255)),
    12: ("ff557f", "UR2", (255, 85, 127)),
    13: ("55aa7f", "UR3", (85, 170, 127)),
    14: ("ff5500", "UR4", (255, 85, 0)),
    15: ("0055ff", "UR5", (0, 85, 255)),
    16: ("aa0000", "UR6", (170, 0, 0)),
    17: ("49f7eb", "UR7", (73, 247, 235)),
    18: ("7d12f7", "UR8", (125, 18, 247)),


    # lower
    31: ("f00000", "LL1", (240, 0, 0)),
    32: ("fbff03", "LL2", (251, 255, 3)),
    33: ("2cfbff", "LL3", (44, 251, 255)),
    34: ("f12fff", "LL4", (241, 47, 255)),
    35: ("7dff9b", "LL5", (125, 255, 155)),
    36: ("1a7dff", "LL6", (26, 125, 255)),
    37: ("ffea9d", "LL7", (255, 234, 157)),
    38: ("cc7e7e", "LL8", (204, 126, 126)),

    41: ("ce81d4", "LR1", (206, 129, 212)),
    42: ("2d8742", "LR2", (45, 135, 66)),
    43: ("b9cf2d", "LR3", (185, 207, 45)),
    44: ("4593cf", "LR4", (69, 147, 207)),
    45: ("cf4868", "LR5", (207, 72, 104)),
    46: ("04cf04", "LR6", (4, 207, 4)),
    47: ("2301cf", "LR7", (35, 1, 207)),
    48: ("52cca9", "LR8", (82, 204, 169)),
}


def cal_metric(gt_labels, pred_labels, target_class=None):
    pred_softmax = torch.nn.functional.softmax(pred_labels, dim=1)
    _, pred_classes = torch.max(pred_softmax, dim=1)

    if target_class is not None:
        gt_labels = (gt_labels == target_class)
        pred_classes = (pred_classes == target_class)

    correct_mask = pred_classes == gt_labels
    wrong_mask = pred_classes != gt_labels
    positive_mask = gt_labels != 0
    negative_mask = gt_labels == 0

    TP = torch.sum(positive_mask * correct_mask).item()
    TN = torch.sum(negative_mask * correct_mask).item()
    FP = torch.sum(positive_mask * wrong_mask).item()
    FN = torch.sum(negative_mask * wrong_mask).item()

    eps = 1e-6

    ACC = (TP + TN) / (FP + TP + FN + TN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    F1 = 2 * (precision * recall) / (precision + recall + eps)
    IOU = TP / (FP + TP + FN + eps)

    return ACC, precision, recall, F1, IOU, pred_classes


def output_pred_ply(pred_mask, cell_coords, path, point_coords=None, face_info=None):
    if point_coords is not None and face_info is not None:
        vertex_info = ""
        cell_info = ""
        for pc in point_coords:
            vertex_info += f'{pc[0]} {pc[1]} {pc[2]} {125} {125} {125} {255}\n'

        valid_face_num = 0
        for color, fi in zip(pred_mask, face_info):
            if fi[0] == 0 and fi[1] == 0 and fi[2] == 0:
                continue
            cell_info += f'3 {int(fi[0])} {int(fi[1])} {int(fi[2])} {color[0]} {color[1]} {color[2]} {255}\n'
            valid_face_num += 1
        header = (f"ply\n"
                  f"format ascii 1.0\n"
                  f"comment VCGLIB generated\n"
                  f"element vertex {point_coords.shape[0]}\n"
                  f"property double x\n"
                  f"property double y\n"
                  f"property double z\n"
                  f"property uchar red\n"
                  f"property uchar green\n"
                  f"property uchar blue\n"
                  f"property uchar alpha\n"
                  f"element face {valid_face_num}\n"
                  f"property list uchar int vertex_indices\n"
                  f"property uchar red\n"
                  f"property uchar green\n"
                  f"property uchar blue\n"
                  f"property uchar alpha\n"
                  f"end_header\n")
    else:
        header = (f"ply\n"
                  f"format ascii 1.0\n"
                  f"comment VCGLIB generated\n"
                  f"element vertex {cell_coords.shape[0]}\n"
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
        cell_info = ""
        for color, coord in zip(pred_mask, cell_coords):
            vertex_info += f'{coord[0]} {coord[1]} {coord[2]} {color[0]} {color[1]} {color[2]} {255}\n'

    with open(path, 'w', encoding='ascii') as f:
        f.write(header)
        f.write(vertex_info)
        f.write(cell_info)

    return