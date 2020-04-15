import torch
import torch.nn as nn
import math
import numpy as np


def bbox_overlaps_aligned(bboxes1, bboxes2, is_aligned=False):
    '''
    Param:
    bboxes1:   FloatTensor(n, 4) # 4: ymin, xmin, ymax, xmax
    bboxes2:   FloatTensor(n, 4)

    Return:    
    FloatTensor(n)
    '''
    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    hw = (br - tl + 1).clamp(min=0)  # [rows, 2]
    overlap = hw[:, 0] * hw[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)
    return ious

def smooth_l1_loss(pred, target, beta=0.11):
    x = (pred - target).abs()
    l1 = x - 0.5 * beta
    l2 = 0.5 * x ** 2 / beta
    return torch.where(x >= beta, l1, l2)

def iou_loss(pred, target, eps=1e-6):
    '''
    Param:
    pred:     FloatTensor(n, 4) # 4: ymin, xmin, ymax, xmax
    target:   FloatTensor(n, 4)

    Return:    
    FloatTensor(n)
    '''
    ious = bbox_overlaps_aligned(pred, target).clamp(min=eps)
    loss = -ious.log()
    return loss

def smooth_l1_loss(pred, target, beta=0.11):
    x = (pred - target).abs()
    l1 = x - 0.5 * beta
    l2 = 0.5 * x ** 2 / beta
    return torch.where(x >= beta, l1, l2)

def bbox_overlaps_diou(bboxes1, bboxes2):

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return torch.sum(1.0 - dious)

def diou_loss(bboxes1, bboxes2):
    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    hw = (br - tl + 1).clamp(min=0)  # [rows, 2]
    overlap = hw[:, 0] * hw[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)

    centerx1 = (bboxes1[:,0]+bboxes1[:,2])/2
    centery1 = (bboxes1[:,1]+bboxes1[:,3])/2
    centerx2 = (bboxes2[:,0]+bboxes2[:,2])/2
    centery2 = (bboxes2[:,1]+bboxes2[:,3])/2

    tl = torch.min(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    hw = (br-tl+1).clamp(min=0)

    outer_diag = hw[:,0]**2 + hw[:,1]**2
    inter_diag = (centerx1-centerx2)**2 + (centery1-centery2)**2
    return 1-ious+ inter_diag/outer_diag

def diou_loss_diy(bboxes1, bboxes2):
    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    hw = (br - tl + 1).clamp(min=0)  # [rows, 2]
    overlap = hw[:, 0] * hw[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)

    centerx1 = (bboxes1[:,0]+bboxes1[:,2])/2
    centery1 = (bboxes1[:,1]+bboxes1[:,3])/2
    centerx2 = (bboxes2[:,0]+bboxes2[:,2])/2
    centery2 = (bboxes2[:,1]+bboxes2[:,3])/2

    tl = torch.min(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    hw = (br-tl+1).clamp(min=0)

    h1 = bboxes1[:,2]-bboxes1[:,0]
    w1 = bboxes1[:,3]-bboxes1[:,1]
    h2 = bboxes2[:,2]-bboxes2[:,0]
    w2 = bboxes2[:,3]-bboxes2[:,1]

    wx = smooth_l1_loss(w1,w2)
    hx = smooth_l1_loss(h1,h2)
    outer_diag = hw[:,0]**2 + hw[:,1]**2
    inter_diag = (centerx1-centerx2)**2 + (centery1-centery2)**2
    return 1 - ious + inter_diag/outer_diag + wx + hx

def ciou_loss(bboxes1, bboxes2):
    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    hw = (br - tl + 1).clamp(min=0)  # [rows, 2]
    overlap = hw[:, 0] * hw[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)

    centerx1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2
    centery1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2
    centerx2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2
    centery2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2

    tl = torch.min(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    hw = (br - tl + 1).clamp(min=0)

    h1 = bboxes1[:, 2] - bboxes1[:, 0]
    w1 = bboxes1[:, 3] - bboxes1[:, 1]
    h2 = bboxes2[:, 2] - bboxes2[:, 0]
    w2 = bboxes2[:, 3] - bboxes2[:, 1]

    with torch.no_grad():
        arctan = torch.atan(w1/h1) - torch.atan(w2/h2)
        v = (4 /(math.pi**2)) * torch.pow((torch.atan(w1/h1)-torch.atan(w2/h2)),2)
        S=1-ious
        alpha = v/(S+v)
        w_temp = 2 * w1
    ar = (8 / (math.pi**2)) *arctan*((w1-w_temp)*h1)
    outer_diag = hw[:, 0] ** 2 + hw[:, 1] ** 2
    inter_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
    u = inter_diag /outer_diag
    cious = ious - (u + alpha * ar)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    return 1 - cious

def diou_loss_diy2(bboxes1, bboxes2):
    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    hw = (br - tl + 1).clamp(min=0)  # [rows, 2]
    overlap = hw[:, 0] * hw[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)
    l1_loss = smooth_l1_loss(bboxes1, bboxes2)
    log = [(1-ious).mean().data.cpu().numpy(), l1_loss.mean().data.cpu().numpy()]
    np.save('./bifpn_weight/log_loss.npy', log)
    return 1 - ious + 2.5 * l1_loss.sum(dim=1)




