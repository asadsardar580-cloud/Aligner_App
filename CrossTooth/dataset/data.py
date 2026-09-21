import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from dataset import data_util
import vedo
import argparse

class ToothData(Dataset):
    def __init__(self, args):
        self.args = args
        self.point_transform = transforms.Compose(
            [
                data_util.PointcloudToTensor(),
                data_util.PointcloudNormalize(radius=1),
                data_util.PointcloudSample(total=args.num_points, sample=args.sample_points)
            ]
        )

    def get_by_name(self, name):
        return self.__getitem__(name)

    def __getitem__(self, item):
        mesh = vedo.load(item)
        cell_normals = np.array(mesh.normals(cells=True))
        point_coords = np.array(mesh.points())
        face_info = np.array(mesh.cells())
        cell_coords = np.array([[
            (point_coords[point_idxs[0]][0] + point_coords[point_idxs[1]][0] + point_coords[point_idxs[2]][0]) / 3,
            (point_coords[point_idxs[0]][1] + point_coords[point_idxs[1]][1] + point_coords[point_idxs[2]][1]) / 3,
            (point_coords[point_idxs[0]][2] + point_coords[point_idxs[1]][2] + point_coords[point_idxs[2]][2]) / 3,
        ] for point_idxs in mesh.cells()])

        pointcloud = np.concatenate((cell_coords, cell_normals), axis=1)

        if pointcloud.shape[0] < self.args.num_points:
            padding = np.zeros((self.args.num_points - pointcloud.shape[0], pointcloud.shape[1]))
            face_info = np.concatenate((face_info, np.zeros(shape=(self.args.num_points - pointcloud.shape[0], 3))), axis=0)
            pointcloud = np.concatenate((pointcloud, padding), axis=0)

        permute = np.random.permutation(self.args.num_points)
        pointcloud = pointcloud[permute]
        face_info = face_info[permute]

        pointcloud, face_info = self.point_transform([pointcloud, face_info])

        return pointcloud.to(torch.float), point_coords, face_info


if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--case', type=str, default="")
        parser.add_argument('--num_points', type=int, default=16000)
        parser.add_argument('--sample_points', type=int, default=16000)
        args = parser.parse_args()
        return args

    args = get_args()
    dataset = ToothData(args)
    pointcloud, point_coords, face_info = dataset.get_by_name(args.case)
    print(pointcloud.shape)