import numpy as np
from sklearn.metrics import roc_auc_score 

class Evaluator(object):
    def __init__(self, num_class, selective):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)
        self.selective = selective # (N, H, W)

    def _generate_matrix(self, label, pred, selection=None):
        """
        label: annotated mask, numpy.ndarray, [0, 1, ..., (self.num_class-1)], (N, H, W)
        pred: prediction mask, numpy.ndarray, [0, 1, ..., (self.num_class-1)], (N, H, W)
        selection: selected mask, numpy.ndarray, [0, 1], (N, H, W)
        """
        mask = (label >= 0) & (label < self.num_class)
        if self.selective:
            mask = mask & (selection == 1)
        label = self.num_class * label[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength = self.num_class*2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    
    def add_batch(self, label, pred, selection=None):
        assert label.shape == pred.shape # (N, H, W)
        self.confusion_matrix += self._generate_matrix(label, pred, selection=selection)
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)

    def Confusion_Matrix(self):
        print(self.confusion_matrix)
        return self.confusion_matrix

    def get_Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def get_Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def get_Pixel_Accuracy_Class_S(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return Acc

    def get_Precision(self):
        Prec = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        return Prec
    
    def get_Recall(self):
        Recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return Recall

    def get_F1_Score(self, Prec, Recall):
        F1_score = 2 * (Prec * Recall) / (Prec + Recall)
        return F1_score

    def get_mIoU(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def get_IoU_Class(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return MIoU

    def get_FWIoU(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
   
    def get_Dice_Score(self):
        dice_score = 2*np.diag(self.confusion_matrix)/(np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0))
        return dice_score



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