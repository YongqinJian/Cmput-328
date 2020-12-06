import time
import numpy as np
from skimage.draw import polygon

from A6_submission import classify_and_detect


def compute_classification_acc(pred, gt):
    assert pred.shape == gt.shape
    return (pred == gt).astype(int).sum() / gt.size


def compute_iou(b_pred, b_gt):
    """

    :param b_pred: predicted bounding boxes, shape=(n,2,4)
    :param b_gt: ground truth bounding boxes, shape=(n,2,4)
    :return:
    """

    n = np.shape(b_gt)[0]
    L_pred = np.zeros((64, 64))
    L_gt = np.zeros((64, 64))
    iou = 0.0
    for i in range(n):
        for b in range(2):
            rr, cc = polygon([b_pred[i, b, 0], b_pred[i, b, 0], b_pred[i, b, 2], b_pred[i, b, 2]],
                             [b_pred[i, b, 1], b_pred[i, b, 3], b_pred[i, b, 3], b_pred[i, b, 1]], [64, 64])
            L_pred[rr, cc] = 1

            rr, cc = polygon([b_gt[i, b, 0], b_gt[i, b, 0], b_gt[i, b, 2], b_gt[i, b, 2]],
                             [b_gt[i, b, 1], b_gt[i, b, 3], b_gt[i, b, 3], b_gt[i, b, 1]], [64, 64])
            L_gt[rr, cc] = 1

            iou += (1.0 / (2 * n)) * (np.sum((L_pred + L_gt) == 2) / np.sum((L_pred + L_gt) >= 1))

            L_pred[:, :] = 0
            L_gt[:, :] = 0

    return iou


def main():
    # prefix = "test"
    prefix = "valid"

    images = np.load(prefix + "_X.npy")

    start_t = time.time()
    pred_class, pred_bboxes = classify_and_detect(images)
    end_t = time.time()

    gt_class = np.load(prefix + "_Y.npy")
    gt_bboxes = np.load(prefix + "_bboxes.npy")
    acc = compute_classification_acc(pred_class, gt_class)
    iou = compute_iou(pred_bboxes, gt_bboxes)

    time_taken = end_t - start_t

    print(f"Classification Acc: {acc}")
    print(f"Detection IOU: {iou}")
    print(f"Test time: {time_taken}")


if __name__ == '__main__':
    main()
