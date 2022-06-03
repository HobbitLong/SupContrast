from __future__ import print_function

import math
import cv2
import numpy as np
import torch
import torch.optim as optim


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def color_masking(img, rgb):
    r, g, b = rgb
    return np.logical_and(np.logical_and(img[:, :, 0] == r, img[:, :, 1] == g), img[:, :, 2] == b)


def logical_or_masks(mask_list):
    mask_all = np.zeros_like(mask_list[0], dtype=bool)
    for mask in mask_list:
        mask_all = np.logical_or(mask_all, mask)
    return mask_all


def seg2mask(seg):
    img_numpy = np.array(seg)

    m_background = color_masking(img_numpy, (0, 0, 0))
    m_skin = color_masking(img_numpy, (204, 0, 0))
    m_nose = color_masking(img_numpy, (76, 153, 0))
    m_eye_g = color_masking(img_numpy, (204, 204, 0))
    m_l_eye = color_masking(img_numpy, (51, 51, 255))
    m_r_eye = color_masking(img_numpy, (204, 0, 204))
    m_l_brow = color_masking(img_numpy, (0, 255, 255))
    m_r_brow = color_masking(img_numpy, (255, 204, 204))
    m_l_ear = color_masking(img_numpy, (102, 51, 0))
    m_r_ear = color_masking(img_numpy, (255, 0, 0))
    m_mouth = color_masking(img_numpy, (102, 204, 0))
    m_u_lip = color_masking(img_numpy, (255, 255, 0))
    m_l_lip = color_masking(img_numpy, (0, 0, 153))
    m_hair = color_masking(img_numpy,  (0, 0, 204))
    m_hat = color_masking(img_numpy, (255, 51, 153))
    m_ear_r = color_masking(img_numpy, (0, 204, 204))
    m_neck_l = color_masking(img_numpy, (0, 51, 0))
    m_neck = color_masking(img_numpy, (255, 153, 51))
    m_cloth = color_masking(img_numpy, (0, 204, 0))

    # gen mask for using in the model
    mask_face = logical_or_masks([m_skin, m_l_ear, m_r_ear])
    mask_hair = logical_or_masks([m_hair, m_hat])
    mask_eye = logical_or_masks([m_l_brow, m_r_brow, m_l_eye, m_r_eye, m_eye_g])
    mask_nose = logical_or_masks([m_nose])
    mask_lip = logical_or_masks([m_u_lip, m_l_lip])
    mask_tooth = logical_or_masks([m_mouth])
    mask_head = logical_or_masks([m_skin, m_l_ear, m_r_ear, m_l_brow, m_r_brow, m_l_eye, 
                                    m_r_eye, m_eye_g, m_nose, m_u_lip, m_l_lip, m_mouth])
    mask_background = logical_or_masks([m_background, m_cloth, m_hat, m_hair])
    
    # merge masks 
    masks = np.array([mask_face, mask_eye, mask_nose, mask_lip, mask_tooth])
    mask_head = np.array([mask_head])
    mask_background = np.array([mask_background])
    return masks, mask_head, mask_background


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.Adam(model.parameters(),
                          lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
