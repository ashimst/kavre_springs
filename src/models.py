"""
Standard classifiers used in the benchmark, with fixed hyperparameters (no tuning on test data).

logistic regression (L2, C = 1, class-balanced), random forest (300 trees, depth 12, class-balanced) and CatBoost
(400 iterations, depth 6, learning rate 0.05, class-balanced), each seeded independently; and distance-weighted kNN
label interpolation, the baseline for any method that exploits spatial autocorrelation of labels.
"""

import numpy as np
from catboost import CatBoostClassifier
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_models(seed: int = 42, pos_weight: float = 1.0):
    return {
        "Logistic Regression": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                                          random_state=seed),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced",
                                                        random_state=seed, n_jobs=-1),
        "CatBoost": lambda: CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                               auto_class_weights="Balanced", verbose=0, random_seed=seed,
                                               thread_count=-1, allow_writing_files=False),
    }


def knn_label_predict(coords_m: np.ndarray, y: np.ndarray, labelled: np.ndarray, k: int = 7) -> np.ndarray:
    """Inverse-distance-weighted mean label of the k nearest labelled sources, for every source."""
    idx_l = np.where(labelled)[0]
    d, j = cKDTree(coords_m[idx_l]).query(coords_m, k=k)
    w = 1.0 / np.maximum(d, 1.0)
    return (w * y[idx_l][j]).sum(1) / w.sum(1)
