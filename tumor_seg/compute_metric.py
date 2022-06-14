import numpy as np
from sklearn.metrics import roc_auc_score 

def get_target_value_index(array, target_value):
    # index = set([i for i, v in enumerate(array) if v == target_value])
    index = np.where(array == target_value)[0]
    return index

def get_performance(label, output, predict, isprint = False):
    """
    compute pixel-level classification performance from patch-level label and outputs

    shape as (height, width)

    Args
        label: 0 or 255, , np.uint8
        output: ... 0.5 ..., np.float64 or np.float32
        predict: 0 or 255, np.uint8, (output > 0.5)*255

    Variables
        C1: label, tumor class
        C0: label, benign class
        P1: prediction as tumor
        P0: prediction as benign  

        TP: True Positive, right tumor prediction
        TN: True Negative, right benign prediction
        FP: False Positive, wrong tumor prediction
        FN: False Negative, wrong benign prediction

    Returns 
        accuracy: (#TP + #TN) / (Total)
        recall: (#TP) / #C1
        precision: #TP / #P1
        f1_score: 2*recall*precision/(recall+precision)
        auc_score: Area Under Curve in ROC (Receiver operating characteristic) curve
    """

    label = label.flatten()
    output = output.flatten()
    predict = predict.flatten()

    C1, C0 = get_target_value_index(label, 1), get_target_value_index(label, 0)
    P1, P0 = get_target_value_index(predict, 1), get_target_value_index(predict, 0)

    TP, TN = np.intersect1d(C1, P1), np.intersect1d(C0, P0) 
    FP, FN = np.setdiff1d(P1, C1), np.setdiff1d(P0, C0)

    accuracy = (len(TP) + len(TN))/(len(C1) + len(C0))

    recall, precision, f1_score = np.NaN, np.NaN, np.NAN
    if len(C1) != 0:    recall = len(TP) / len(C1)
    if len(P1) != 0:    precision = len(TP) / len(P1)

    if recall != np.NaN and precision != np.NaN and (recall+precision) !=0:
        f1_score = 2*recall*precision/(recall + precision)

    try: auc_score = roc_auc_score(label, output)
    except: auc_score = np.NaN

    if isprint:
        print(f'accuracy: {accuracy:.3f} | recall: {recall:.3f} | precision: {precision:.3f} | f1 score: {f1_score:.3f} | AUC score: {auc_score:.3f}')

    return accuracy, recall, precision, f1_score, auc_score

def compute_IoU(label, pred, index, EPS = 1e-6):
    temp_label = np.zeros_like(label, dtype=np.uint8)
    temp_pred = np.zeros_like(pred, dtype=np.uint8)

    temp_label[label == index] = 1
    temp_pred[pred == index] = 1

    intersect = float((temp_label & temp_pred).sum())
    union = float((temp_label | temp_pred).sum())
    iou = (intersect + EPS) / (union + EPS)

    return iou

def compute_mIOU(label, pred, n_class = 2):
    miou = 0
    for i in range(n_class):
        miou += compute_IoU(label, pred, index = i)
    miou /= float(n_class)
    return miou

if __name__ == "__main__":
    # label = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    # output = np.array([0.9, 0.8, 0.7, 0.9, 0.7, 0.8, 0.1, 0.2])
    # predict = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    # get_performance(label, output, predict, isprint=True)


    # label = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
    # pred = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])

    # intersect = float((label & pred).sum())
    # union = float((label | pred).sum())

    # print(intersect, union)

    import os

    data_dir = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_baseline_samsung_data/1-fold/output/model_pred'
    # data_dir = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/01_5-f_cv_baseline/minmax/model_pred'
    label_list = [l for l in sorted(os.listdir(data_dir)) if 'label' in l]
    pred_list  = [p for p in sorted(os.listdir(data_dir)) if 'pred' in p]

    assert len(label_list) == len(pred_list), 'numbers of label and prediction does not match'

    N = len(label_list)

    from PIL import Image
    from tqdm import tqdm

    IoU_0, IoU_1 = 0, 0
    mIoU = 0

    for (l, p) in tqdm(zip(label_list, pred_list), total = N):
        if l.split('label')[0] != p.split('pred')[0]:
            print(f'Required prediction corresponding to label | label: {l}, pred: {p}')
            break
        
        label = np.array(Image.open(os.path.join(data_dir, l)).convert('L'))
        pred = np.array(Image.open(os.path.join(data_dir, p)).convert('L'))

        # if not (0 in set(list(np.ravel(label))) or 255 in set(list(np.ravel(label)))):
        #     print(l)

        # if not (0 in set(list(np.ravel(pred))) or 255 in set(list(np.ravel(pred)))):
        #     print(l)

        label = np.uint8(label/255)
        pred = np.uint8(pred/255)

        iou_0 = compute_IoU(label, pred, 0)
        iou_1 = compute_IoU(label, pred, 1)

        miou = (iou_0 + iou_1) / 2.

        IoU_0 += iou_0
        IoU_1 += iou_1
        mIoU += miou

    IoU_0 /= N
    IoU_1 /= N
    mIoU /= N

    print('IoU (benign)', IoU_0)
    print('IoU (tumor)', IoU_1)
    print('Mean IoU', mIoU)