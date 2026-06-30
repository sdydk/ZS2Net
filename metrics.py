import cv2
import numpy as np

def iou(predicted, target, n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = predicted == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious

def pixel_acc(predicted, target):
    correct = (predicted == target).sum()
    total = (target == target).sum()
    return correct / total

# Dice
def dice_coeff(predicted, target):
    smooth = 1.
    num = predicted.size(0)
    m1 = predicted.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def two_calculate_metrics(pred, target):
    pred_img = np.array(pred, dtype=bool)
    gt_img = np.array(target, dtype=bool)

    TP = np.sum(np.logical_and(pred_img, gt_img))
    TN = np.sum(np.logical_and(np.logical_not(pred_img), np.logical_not(gt_img)))
    FP = np.sum(np.logical_and(pred_img, np.logical_not(gt_img)))
    FN = np.sum(np.logical_and(np.logical_not(pred_img), gt_img))

    IOU = TP / (TP + FN + FP + 1e-7)
    DICE = 2 * TP / (2 * TP + FN + FP + 1e-7)
    ACCURACY = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    PRECISION = TP / (TP + FP + 1e-7)
    RECALL = TP / (TP + FN + 1e-7)
    SENSITIVITY = TP / (TP + FN + 1e-7)
    F1_SCORE = 2*(PRECISION*RECALL) / (PRECISION + RECALL + 1e-7)
    SPECIFICITY = TN / (TN + FP + 1e-7)

    return IOU, DICE, ACCURACY, PRECISION, RECALL, SENSITIVITY, F1_SCORE, SPECIFICITY
################################################################################
################################################################################
def ConfusionMatrix(pred, target, n_class):
    mask = (pred >= 0) & (pred < n_class)
    label = n_class * pred[mask].astype(int) + target[mask]
    count = np.bincount(label, minlength=n_class**2)
    confusionmatrix = count.reshape(n_class, n_class)
    return confusionmatrix
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2
    ).reshape(n_class, n_class)

def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum() #以一维数组的形式返回方阵的对角线元素
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist).sum() / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    m_iou = np.nanmean(iou)
    freq = hist.sum(axis=1) / hist.sum()

    return acc, acc_cls, m_iou
def miou(confusionmatrix):
    intersection = np.diag(confusionmatrix) #TP

    # axis=1,按行,axis=0按列

    union = np.sum(confusionmatrix, axis=1) + np.sum(confusionmatrix, axis=0) - np.diag(confusionmatrix) # TP+FP+FN
    IoU = intersection / (union + 1e-7)
    mIoU = np.nanmean(IoU)
    return mIoU

def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h +1, 1 : w + 1]
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.005, cls_num=2):

    # gt = gt[0, 0]
    # dt = dt[0]
    # gt = gt.astype(np.uint8)
    # dt = dt.astype(np.uint8)
    # gt = gt.numpy().astype(np.uint8)
    # dt = dt.numpy().astype(np.uint8)
    boundary_iou_list = []
    for i in range(cls_num):
        gt_i = (gt == i).astype(np.uint8)
        dt_i = (dt == i).astype(np.uint8)
        gt_boundary = mask_to_boundary(gt_i, dilation_ratio)
        dt_boundary = mask_to_boundary(dt_i, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        if union < 1:
            boundary_iou_list.append(0)
            continue
        boundary_iou = intersection / union
        boundary_iou_list.append(boundary_iou)
    return np.array(boundary_iou_list)
if __name__ == '__main__':
    t1 = np.array([0, 0, 1, 1, 2, 2])
    t2 = np.array([0, 0, 1, 1, 2, 2])
    result = ConfusionMatrix(t1, t2, 3)
    pixel_acc1 = np.diag(result).sum() / result.sum()
    mIoU1 = miou(result)



    img_1 = cv2.imread(r'D:\PycharmProjects\Segmentation\dataset\zooplankton\Arthropoda\Amphipoda\Gammaridean1.tif')
    img_2 = cv2.imread(r'D:\PycharmProjects\Segmentation\dataset\zooplankton\Arthropoda\Amphipoda\Gammaridean2.tif')
    two_calculate_metrics(img_1, img_2)
    print("--------------------")