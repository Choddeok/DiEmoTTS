import numpy as np
from operator import itemgetter
from sklearn import metrics


def cosine_similarity(x, y):
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return (tunedThreshold, eer, fpr, fnr)
