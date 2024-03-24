import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score


def score(logits, labels):
    prediction_score = logits.detach().cpu().numpy()
    labels = labels.cpu().numpy()

    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        prediction_label = np.where(prediction_score > threshold, 1, 0)
        f1 = f1_score(labels, prediction_label)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    prediction_label = np.where(prediction_score > best_threshold, 1, 0)
    accuracy = (prediction_label == labels).sum() / len(prediction_label)
    auroc = roc_auc_score(labels, prediction_score)
    auprc = average_precision_score(labels, prediction_score)
    precision = precision_score(labels, prediction_label)
    recall = recall_score(labels, prediction_label)

    return accuracy, auroc, auprc, best_f1, precision, recall
