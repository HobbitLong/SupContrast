import torch


def cos_sim(x1, x2):
    return torch.sum(x1 * x2, dim=0) / (torch.norm(x1, dim=0) * torch.norm(x2, dim=0))
