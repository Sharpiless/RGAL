import torch
import torch.nn as nn
import torch.nn.functional as F

BIG_NUMBER = 1e12


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def pos_neg_mask(labels):
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) * \
               (1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device))
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) * \
               (1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device))

    return pos_mask, neg_mask

class BalancedWeighted(nn.Module):
    cut_off = 0.5
    nonzero_loss_cutoff = 1.0
    """
    Distance Weighted loss assume that embeddings are normalized py 2-norm.
    """

    def __init__(self, dist_func=pdist):
        self.dist_func = dist_func
        super().__init__()

    def forward(self, embeddings, labels):
        with torch.no_grad():
            embeddings = F.normalize(embeddings, dim=1, p=2)
            pos_mask, neg_mask = pos_neg_mask(labels)
            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            d = embeddings.size(1)
            dist = (pdist(embeddings, squared=True) + torch.eye(embeddings.size(0),
                                                                device=embeddings.device, dtype=torch.float32)).sqrt()
            log_weight = ((2.0 - d) * dist.log() - ((d - 3.0)/2.0)
                          * (1.0 - 0.25 * (dist * dist)).log())
            weight = (log_weight - log_weight.max(dim=1,
                                                  keepdim=True)[0]).exp()
            weight = weight * \
                (neg_mask * (dist < self.nonzero_loss_cutoff) * (dist > self.cut_off)).float()

            weight = weight + \
                ((weight.sum(dim=1, keepdim=True) == 0) * neg_mask).float()
            weight = weight / (weight.sum(dim=1, keepdim=True))
            weight = weight[anchor_idx]
            if torch.max(weight) == 0:
                return None, None, None
            weight[torch.isnan(weight)] = 0
            neg_idx = torch.multinomial(weight, 1).squeeze(1)

        return anchor_idx, pos_idx, neg_idx


class DistanceWeighted(nn.Module):
    cut_off = 0.5
    nonzero_loss_cutoff = 1.4
    """
    Distance Weighted loss assume that embeddings are normalized py 2-norm.
    """

    def __init__(self, dist_func=pdist):
        self.dist_func = dist_func
        super().__init__()

    def forward(self, embeddings, labels):
        with torch.no_grad():
            embeddings = F.normalize(embeddings, dim=1, p=2)
            pos_mask, neg_mask = pos_neg_mask(labels)
            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            d = embeddings.size(1)
            dist = (pdist(embeddings, squared=True) + torch.eye(embeddings.size(0),
                                                                device=embeddings.device, dtype=torch.float32)).sqrt()
            dist = dist.clamp(min=self.cut_off)
            log_weight = ((2.0 - d) * dist.log() - ((d - 3.0)/2.0)
                          * (1.0 - 0.25 * (dist * dist)).log())
            weight = (log_weight - log_weight.max(dim=1,
                                                  keepdim=True)[0]).exp()
            weight = weight * \
                (neg_mask * (dist < self.nonzero_loss_cutoff)).float()

            weight = weight + \
                ((weight.sum(dim=1, keepdim=True) == 0) * neg_mask).float()
            weight = weight / (weight.sum(dim=1, keepdim=True))
            weight = weight[anchor_idx]
            if torch.max(weight) == 0:
                return None, None, None
            weight[torch.isnan(weight)] = 0
            neg_idx = torch.multinomial(weight, 1).squeeze(1)

        return anchor_idx, pos_idx, neg_idx


class TriLoss(nn.Module):
    def __init__(self, p=2, margin=0.2, balanced_sampling=False):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        if balanced_sampling:
            self.sampler = BalancedWeighted()
        else:
            self.sampler = DistanceWeighted()
        self.sampler.dist_func = lambda e: pdist(e, squared=(p == 2))
        self.count = 0

    def forward(self, stu_features, logits, labels, negative=False):
        if negative:
            anchor_idx, neg_idx, pos_idx = self.sampler(logits, labels)
        else:
            anchor_idx, pos_idx, neg_idx = self.sampler(logits, labels)

        loss = 0.
        if anchor_idx is None:
            print('warning: no negative samples found.')
            return torch.zeros(1)
        self.count += 1
        for embeddings in stu_features:
            if len(embeddings.shape) > 2:
                embeddings = embeddings.mean(dim=(2, 3), keepdim=False)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            anchor_embed = embeddings[anchor_idx]
            positive_embed = embeddings[pos_idx]
            negative_embed = embeddings[neg_idx]

            triloss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                            margin=self.margin, p=self.p, reduction='none')
            loss += triloss

        return loss.mean()


def prune_fpgm(layers):

    pruned_activations_mask = []
    with torch.no_grad():

        for layer in layers:

            b, c, h, w = layer.shape

            P = layer.view((b, c, h * w))

            A = P @ P.transpose(1, 2)
            A = torch.sum(A, dim=-1)
            max_ = torch.max(A)
            min_ = torch.min(A)
            A = 1.0 - (A - min_) / (max_ - min_) + 1e-3

            pruned_activations_mask.append(A.to(layer.device))

    return pruned_activations_mask


class CDLoss(nn.Module):
    """Channel Distillation Loss"""

    def __init__(self, linears=[]):
        super().__init__()
        self.linears = linears

    def forward(self, stu_features: list, tea_features: list):
        loss = 0.
        for i, (s, t) in enumerate(zip(stu_features, tea_features)):
            if not self.linears[i] is None:
                s = self.linears[i](s)
            s = s.mean(dim=(2, 3), keepdim=False)
            t = t.mean(dim=(2, 3), keepdim=False)
            # loss += F.mse_loss(F.normalize(s, p=2, dim=-1), F.normalize(t, p=2, dim=-1))
            loss += F.mse_loss(s, t)
        return loss


class GRAMLoss(nn.Module):
    """GRAM Loss"""

    def __init__(self, linears=[]):
        super().__init__()
        self.linears = linears

    def forward(self, stu_features: list, tea_features: list):
        loss = 0.
        masks = prune_fpgm(tea_features)
        for i, s in enumerate(stu_features):
            t = tea_features[i]
            if not self.linears[i] is None:
                s = self.linears[i](s)
            b, c = masks[i].shape
            m = masks[i].view((b, c, 1, 1)).detach()
            loss += torch.mean(torch.pow(s - t, 2) * m)
        return loss
