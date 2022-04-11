import numpy as np
from sklearn.metrics import roc_auc_score 

def get_performance(label, output, predict, isprint = False):

    label = label.flatten()
    output = output.flatten()
    predict = predict.flatten()

    def get_target_value_index(array, target_value):
        return set([i for i, v in enumerate(array) if v == target_value])

    C1, C0 = get_target_value_index(label, 1), get_target_value_index(label, 0)
    P1, P0 = get_target_value_index(predict, 1), get_target_value_index(predict, 0)

    TP, TN = C1 & P1, C0 & P0 

    FP, FN = P1 - C1, P0 - C0

    accuracy = (len(TP) + len(TN))/(len(C1) + len(C0))

    recall, precision, recall = np.NaN, np.NaN, np.NAN
    if len(C1) != 0:    recall = len(TP) / len(C1)
    if len(P1) != 0:    precision = len(TP) / len(P1)

    if recall != np.NaN and precision != np.NaN:
        f1_score = 2*recall*precision/(recall + precision)

    try: auc_score = roc_auc_score(label, output)
    except: auc_score = np.NaN

    if isprint:
        print(f'accuracy: {accuracy:.3f} | recall: {recall:.3f} | precision: {precision:.3f} | f1 score: {f1_score:.3f} | AUC score: {auc_score:.3f}')

    return accuracy, recall, precision, f1_score, auc_score

if __name__ == "__main__":
    label = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    output = np.array([0.9, 0.8, 0.7, 0.9, 0.7, 0.8, 0.1, 0.2])
    predict = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    get_performance(label, output, predict, isprint=True)
