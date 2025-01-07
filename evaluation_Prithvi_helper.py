import numpy as np

def compute_miou(preds, targets, num_classes):
    """
    This function calculates the IoU for each class and computes the mean (mIoU).
    Classes with no instances in the batch are ignored using NaN.
    """

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    ious = []
    for c in range(num_classes):
        intersection = np.logical_and(preds == c, targets == c).sum()
        union = np.logical_or(preds == c, targets == c).sum()
        if union == 0:
            ious.append(float('nan'))  # Avoid division by zero
        else:
            ious.append(intersection / union)

    # Compute mean IoU, ignoring NaN values (for classes not present in the batch)
    mean_iou = np.nanmean(ious)
    return ious, mean_iou

def compute_acc(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    correct_pixels = np.sum(preds == targets)
    total_pixels = np.prod(targets.shape)
    accuracy = correct_pixels / total_pixels

    return accuracy

