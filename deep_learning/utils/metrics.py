import numpy as np

class Metrics:
    def __init__(self, labels=None, zero_division=0.0):
        self.labels = None if labels is None else np.asarray(labels)
        self.zero_division = float(zero_division)
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self._y_true.clear()
        self._y_pred.clear()

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        self._y_true.append(y_true)
        self._y_pred.append(y_pred)

    def compute(self):
        y_true = np.concatenate(self._y_true)
        y_pred = np.concatenate(self._y_pred)

        labels = self.labels
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))

        n = len(labels)
        idx = {c: i for i, c in enumerate(labels)}

        cm = np.zeros((n, n), dtype=float)
        for t, p in zip(y_true, y_pred):
            cm[idx[t], idx[p]] += 1

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        precision = np.divide(
            tp, tp + fp,
            out=np.full(n, self.zero_division),
            where=(tp + fp) != 0
        )

        recall = np.divide(
            tp, tp + fn,
            out=np.full(n, self.zero_division),
            where=(tp + fn) != 0
        )

        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.full(n, self.zero_division),
            where=(precision + recall) != 0
        )

        return {
            "confusion_matrix": cm,
            "bacc": recall.mean(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": f1.mean(),
        }