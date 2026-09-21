import torch
from torch import nn
import torch.nn.functional as F
from models.PointTransformer.libs.pointops.functions import pointops


_inf = 1e9
_eps = 1e-12


class CBLLoss(nn.Module):

    def __init__(self, nsample=8, num_classes=17, dist_func='l2', contrast_func='nce'):
        super().__init__()
        # KNN邻居的个数k=16
        self.nsample = nsample
        self.num_classes = num_classes
        assert dist_func in ['l2', 'kl']
        self.dist_func = self.dist_l2 if dist_func == 'l2' else self.dist_kl
        assert contrast_func in ['nce', 'softnn']
        self.contrast_func = self.contrast_nce if contrast_func == 'nce' else self.contrast_softnn
        self.main_contrast = self.point_contrast

        self.dist = 'norm'
        self.temperature = None

    def dist_l2(self, features, neighbor_feature):
        dist = torch.unsqueeze(features, -2) - neighbor_feature
        dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + _eps)  # [m, nsample]
        return dist

    def dist_kl(self, features, neighbor_feature, normalized, normalized_neighbor):
        if normalized in [False, 'softmax']:
            features = F.log_softmax(features, dim=-1)
            log_target = True
        elif normalized == True:
            log_target = False
        else:
            raise ValueError(f'kl dist not support normalized = {normalized}')
        features = features.unsqueeze(-2)

        if normalized_neighbor in [False, 'softmax']:
            neighbor_feature = F.log_softmax(neighbor_feature, dim=-1)
        elif normalized_neighbor == True:
            neighbor_feature = torch.maximum(neighbor_feature, neighbor_feature.new_full([], _eps)).log()
        else:
            raise ValueError(f'kl dist not support normalized_neighbor = {normalized}')

        # (input, target) - i.e. (pred, gt), where input/pred should be in log space
        dist = F.kl_div(neighbor_feature, features, reduction='none', log_target=log_target)
        dist = dist.sum(-1)  # [m, nsample]
        return dist

    def contrast_softnn(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = -torch.log(pos / neg + _eps)
        return loss

    def contrast_nce(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        # each Log term an example; per-pos vs. all negs
        neg = torch.sum(exp * (1 - posmask.int()), axis=-1)  # (m)
        under = exp + neg.unsqueeze(-1)
        loss = (exp / under)[posmask]  # each Log term an example
        loss = -torch.log(loss)
        return loss

    def point_contrast(self, p, x, o, nsample, target):
        # 适合 point transformer
        p, features, o = p, x, o

        # 目标分割标签
        labels = target
        neighbor_idx, _ = pointops.knnquery(nsample, p, p, o, o)  # (m, nsample)

        # exclude self-loop
        nsample = nsample - 1
        neighbor_idx = neighbor_idx[..., 1:].contiguous()
        m = neighbor_idx.shape[0]

        neighbor_label = labels[neighbor_idx.view(-1).long()].view(m, nsample)  # (m, nsample)

        if 'norm' in self.dist or self.dist == 'cos':
            features = F.normalize(features, dim=-1)  # p2-norm

        neighbor_feature = features[neighbor_idx.view(-1).long(), :].view(m, nsample, features.shape[1])

        posmask = labels.unsqueeze(-1) == neighbor_label

        # select only pos-neg co-exists
        point_mask = torch.sum(posmask.int(), -1)  # (m)

        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)

        # 特殊情况
        if not torch.any(point_mask):
            return torch.tensor(.0)

        posmask = posmask[point_mask]
        features = features[point_mask]
        neighbor_feature = neighbor_feature[point_mask]

        dist = self.dist_func(features, neighbor_feature)
        loss = self.contrast_func(dist, posmask)  # (m)

        loss = torch.mean(loss)
        return loss

    def forward(self, output, target):
        p, x, o = output[0], output[1], output[2]
        loss = self.main_contrast(p, x, o, self.nsample, target)
        return loss


if __name__ == '__main__':
    criterion = CBLLoss().cuda()
    p = torch.rand((32000, 3)).cuda()
    x = torch.rand((32000, 17)).cuda()
    o = torch.IntTensor([16000, 32000]).cuda()
    target = torch.randint(0, 17, (32000,)).cuda()
    cbl_loss = criterion([p, x, o], target)
    print(cbl_loss.item())
