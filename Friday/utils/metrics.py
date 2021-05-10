from sklearn.metrics import *

metrics = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'average_precision': average_precision_score,
    'f1' : f1_score,
    'f1_micro' : f1_score,
    'precision' : precision_score,
    'recall' : recall_score,
    'roc_auc' : roc_auc_score,
    'r2' : r2_score,
    'mean_squared_error' : mean_squared_error,
    'mean_absolute_error' : mean_absolute_error,
    'max_error' : max_error,
    'explained_variance' : explained_variance_score
}