import torch
import numpy as np

class PointcloudToTensor(object):
    def __call__(self, data):
        points, face_info = data
        return torch.from_numpy(points).float(), face_info


class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __call__(self, data):
        points, face_info = data
        pc = points.numpy()
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return (torch.from_numpy(pc).float(), face_info)



class PointcloudSample(object):
    def __init__(self, total=16000, sample=10000):
        self.total = total
        self.sample = sample

    def __call__(self, data):
        points, face_info = data
        sample = np.random.permutation(self.total)[:self.sample]
        return (points[sample], face_info[sample])