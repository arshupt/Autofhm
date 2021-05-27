from sklearn.metrics import *

def f1_scorer(y_true, y_pred) :

    score = f1_score(y_true, y_pred, average='macro')
    return score

def precision(y_true, y_pred) :

    score = precision_score(y_true, y_pred, average='macro')
    return score

def recall(y_true, y_pred) :

    score = recall_score(y_true, y_pred, average='macro')
    return score

c_metrics = {
    'accuracy': (accuracy_score, False),
    'balanced_accuracy': (balanced_accuracy_score, False),
    'f1' : (f1_scorer, False),
    'precision' : (precision, False),
    'recall' : (recall, False)
}

r_metrics = {
    'r2' : (r2_score, False),
    'mean_squared_error' : (mean_squared_error, True),
    'mean_absolute_error' : (mean_absolute_error, True),
    'max_error' : (max_error, True),
    'explained_variance' : (explained_variance_score, False)
}