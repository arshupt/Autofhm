from sklearn.metrics import *

def avg_precision_score(y_true, y_pred) :

    score = average_precision_score(y_true, y_pred, average='micro')
    return score

def f1_scorer(y_true, y_pred) :

    score = f1_score(y_true, y_pred, average='micro')
    return score

def precision(y_true, y_pred) :

    score = precision_score(y_true, y_pred, average='micro')
    return score

def recall(y_true, y_pred) :

    score = recall_score(y_true, y_pred, average='micro')
    return score



metrics = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'average_precision': avg_precision_score,
    'f1' : f1_scorer,
    'precision' : precision,
    'recall' : recall,
    'roc_auc' : roc_auc_score,
    'r2' : r2_score,
    'mean_squared_error' : mean_squared_error,
    'mean_absolute_error' : mean_absolute_error,
    'max_error' : max_error,
    'explained_variance' : explained_variance_score
}