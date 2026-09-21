import argparse
import torch
import numpy as np

from models.PTv1.point_transformer_seg import PointTransformerSeg38
from utils import output_pred_ply
from utils import label2color_lower
from dataset.data import ToothData

def do_predict(args):

    device = torch.device("cuda" if args.cuda else "cpu")

    enable_pic_feat = False
    point_model = PointTransformerSeg38(in_channels=6, num_classes=17 + 2, pretrain=False, add_cbl=False, enable_pic_feat=enable_pic_feat).to(device)

    pretrained_dict = torch.load(args.pretrain_model_path)
    point_model.load_state_dict(pretrained_dict)

    lower_palette = np.array(
        [[125, 125, 125]] +
        [[label2color_lower[label][2][0],
          label2color_lower[label][2][1],
          label2color_lower[label][2][2]]
         for label in range(1, 17)], dtype=np.uint8)

    ####################
    # Eval
    ####################
    save_mesh = True

    with torch.no_grad():
        point_model.eval()
        dataset = ToothData(args)

        # intra-oral scan to be predicted
        pointcloud, point_coords, face_info = dataset.get_by_name(args.case)
        pointcloud = pointcloud.unsqueeze(0)
        point_coords = point_coords[None, :]
        face_info = face_info[None, :]

        pointcloud = pointcloud.to(device)
        pointcloud = pointcloud.permute(0, 2, 1).contiguous()

        if enable_pic_feat:
            point_seg_result, edge_seg_result = point_model(pointcloud, None)
        else:
            point_seg_result, edge_seg_result = point_model(pointcloud)

        pred_softmax = torch.nn.functional.softmax(point_seg_result, dim=1)
        _, pred_classes = torch.max(pred_softmax, dim=1)

        if save_mesh:
            pred_mask = pred_classes.squeeze(0).cpu().detach().numpy().astype(np.uint8)
            pred_mask[pred_mask == 17] = 0
            pred_mask[pred_mask == 18] = 0
            pred_mask = lower_palette[pred_mask]
            output_pred_ply(pred_mask, None, args.save_path, point_coords[0], face_info[0])

    print(f"Predict end, result at {args.save_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default="YBSESUN6_upper.ply")
    parser.add_argument('--save_path', type=str, default="YBSESUN6_upper_mask.ply")
    parser.add_argument('--pretrain_model_path', type=str, default='models/PTv1/point_best_model.pth')
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--num_points', type=int, default=16000)
    parser.add_argument('--sample_points', type=int, default=16000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    seed = 1
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    do_predict(args)
