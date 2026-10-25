"""Class-count-independent confusion matrices and mIoU."""
import numpy as np


def confusion(gt, pred, num_classes=21):
    gt, pred = np.asarray(gt), np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError('Prediction and mask shapes differ.')
    if not np.isin(gt, [*range(num_classes), 255]).all():
        raise ValueError('Mask values do not follow the contiguous dataset label mapping.')
    valid = (gt >= 0) & (gt < num_classes)
    if ((pred[valid] < 0) | (pred[valid] >= num_classes)).any():
        raise ValueError('Predicted labels are outside the configured class range.')
    return np.bincount(num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metric(hist):
    denom = hist.sum(0) + hist.sum(1) - hist.diagonal()
    iou = np.divide(hist.diagonal(), denom, out=np.zeros(hist.shape[0], dtype=float), where=denom > 0)
    if not (hist.sum(1) > 0).all():
        raise ValueError('All configured classes must be represented in full validation.')
    return {'miou_percent': float(iou.mean() * 100), 'iou_percent': (iou * 100).tolist(), 'confusion': hist.tolist()}
