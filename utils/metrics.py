import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes=2, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        if isinstance(preds,   torch.Tensor): preds   = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()
        preds   = preds.flatten().astype(np.int64)
        targets = targets.flatten().astype(np.int64)
        valid   = (targets >= 0) & (targets < self.num_classes)
        for t, p in zip(targets[valid], preds[valid]):
            self.confusion_matrix[t, p] += 1

    def compute(self):
        cm = self.confusion_matrix.astype(np.float64)
        C  = self.num_classes
        iou = np.zeros(C); precision = np.zeros(C)
        recall = np.zeros(C); f1 = np.zeros(C)
        for c in range(C):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            iou[c]       = tp / (tp + fp + fn + 1e-8)
            precision[c] = tp / (tp + fp + 1e-8)
            recall[c]    = tp / (tp + fn + 1e-8)
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + 1e-8)
        acc = np.diag(cm).sum() / (cm.sum() + 1e-8)
        return {
            "iou":       {self.class_names[c]: float(iou[c])       for c in range(C)},
            "miou":      float(iou.mean()),
            "precision": {self.class_names[c]: float(precision[c]) for c in range(C)},
            "recall":    {self.class_names[c]: float(recall[c])    for c in range(C)},
            "f1":        {self.class_names[c]: float(f1[c])        for c in range(C)},
            "accuracy":  float(acc),
        }

    def print_report(self, results=None):
        if results is None:
            results = self.compute()
        print(f"\n{chr(9472)*60}")
        print(f"  Accuracy: {results['accuracy']:.4f}   mIoU: {results['miou']:.4f}")
        print(f"  {'Class':<12} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        for c in self.class_names:
            print(f"  {c:<12} {results['iou'][c]:>8.4f} "
                  f"{results['precision'][c]:>10.4f} "
                  f"{results['recall'][c]:>8.4f} "
                  f"{results['f1'][c]:>8.4f}")


def compute_bark_area(pts, labels, surface_area_m2=None):
    n_bark  = int((labels == 1).sum())
    n_wood  = int((labels == 0).sum())
    n_total = n_bark + n_wood
    if n_total == 0:
        return {k: 0 for k in ["bark_fraction","bark_area_m2","wood_area_m2",
                                "total_area_m2","n_bark_points","n_wood_points"]}
    bark_fraction = n_bark / n_total
    if surface_area_m2 is None:
        try:
            from scipy.spatial import ConvexHull
            surface_area_m2 = ConvexHull(pts[:, :3]).area
        except Exception:
            xyz = pts[:, :3]
            r = np.sqrt((xyz[:,0]**2 + xyz[:,1]**2)).max()
            h = xyz[:,2].max() - xyz[:,2].min()
            surface_area_m2 = 2 * np.pi * r * h
    return {
        "bark_fraction": float(bark_fraction),
        "bark_area_m2":  float(bark_fraction * surface_area_m2),
        "wood_area_m2":  float((1 - bark_fraction) * surface_area_m2),
        "total_area_m2": float(surface_area_m2),
        "n_bark_points": n_bark,
        "n_wood_points": n_wood,
    }
