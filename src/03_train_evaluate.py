#!/usr/bin/env python3
"""Train three classifiers on the PAM50 labels and see which wins.

Plain setup: stratified 80/20 split, 5-fold CV on the training part, then
final numbers on the held-out test set. The scaler lives inside the
pipeline so each fold only ever sees its own training data.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

import config
from utils import load_processed, save_fig, setup_logging

log = logging.getLogger(__name__)


def make_models():
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=config.RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                           class_weight="balanced_subsample",
                                           random_state=config.RANDOM_STATE)),
        ]),
        "hist_gb": Pipeline([
            ("scaler", StandardScaler()),
            # no class_weight on this one: with it, HGB's sample-weight path
            # makes each fit ~4-5x slower and 5-fold CV impractical on this
            # box. Imbalance is moderate (231/127/98/58), and macro-F1 is
            # reported, so the comparison stays honest.
            ("clf", HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)),
        ]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=config.BASE_DIR)
    args = p.parse_args()
    setup_logging()

    results = args.base_dir / "results"
    results.mkdir(parents=True, exist_ok=True)

    X, genes, ydf = load_processed(args.base_dir / "data" / "processed")
    y = ydf["subtype"].values
    classes = sorted(pd.unique(y))
    log.info("%d samples, %d genes, classes %s", *X.shape, classes)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)

    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    out = {"classes": classes, "n_train": len(ytr), "n_test": len(yte), "models": {}}
    roc_curves, best_auc, best_name = {}, -1.0, ""

    for name, model in make_models().items():
        # folds run sequentially on purpose: parallel folds on top of HGB's
        # own threads just oversubscribes the machine and slows everything down
        cv_acc = cross_val_score(model, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        # predict_proba columns follow model.classes_, reorder to ours
        proba = model.predict_proba(Xte)[:, [list(model.classes_).index(c) for c in classes]]

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        auc = roc_auc_score(label_binarize(yte, classes=classes), proba,
                            average="macro", multi_class="ovr")
        report = classification_report(yte, pred, output_dict=True, zero_division=0)
        out["models"][name] = {
            "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
            "cv_accuracy_std": round(float(cv_acc.std()), 4),
            "test_accuracy": round(float(acc), 4),
            "test_f1_macro": round(float(f1), 4),
            "test_roc_auc_ovr": round(float(auc), 4),
            "per_class": {c: {k: round(float(report[c][k]), 4)
                              for k in ("precision", "recall", "f1-score")}
                          for c in classes},
            "confusion_matrix": confusion_matrix(yte, pred, labels=classes).tolist(),
        }
        log.info("%-13s cv=%.4f±%.4f  test acc=%.4f f1=%.4f auc=%.4f",
                 name, cv_acc.mean(), cv_acc.std(), acc, f1, auc)

        yb = label_binarize(yte, classes=classes)
        roc_curves[name] = {}
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
            roc_curves[name][c] = (fpr, tpr, roc_auc_score(yb[:, i], proba[:, i]))
        if auc > best_auc:
            best_auc, best_name = auc, name

    out["best_model"] = best_name
    (results / "metrics.json").write_text(json.dumps(out, indent=2))

    cm = np.array(out["models"][best_name]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"confusion matrix — {best_name} (test n={len(yte)})")
    save_fig(fig, results / "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for c, (fpr, tpr, a) in roc_curves[best_name].items():
        ax.plot(fpr, tpr, label=f"{c} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC (one-vs-rest) — {best_name}")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, results / "roc_curves.png")

    log.info("best: %s (auc %.4f) — metrics + figures in results/", best_name, best_auc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
