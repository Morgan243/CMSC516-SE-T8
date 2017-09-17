from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred, average):
    res = dict(
        acc=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, average=average),

        precision=precision_score(y_true, y_pred, average=average),
        recall=recall_score(y_true, y_pred, average=average)
        )
    return res


def binary_classification_metrics(y_true, y_pred):
    return classification_metrics(y_true, y_pred, 'binary')


def multi_classification_metrics(y_true, y_pred):
    return classification_metrics(y_true, y_pred, 'weighted')
