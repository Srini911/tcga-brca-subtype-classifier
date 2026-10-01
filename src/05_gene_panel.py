#!/usr/bin/env python3
"""How small can the gene panel get before accuracy falls apart?

A 3000-gene signature is a research result, not something you can run in
a clinic. This ranks genes by ANOVA F-value on the training set only, then
retrains logistic regression on the top-k genes for a few panel sizes and
reports what the test set says. If 20 genes do nearly as well as 3000,
that's the interesting outcome.
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

import config
from utils import load_processed, save_fig, setup_logging

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=config.BASE_DIR)
    args = p.parse_args()
    setup_logging()

    results = args.base_dir / "results"
    X, genes, ydf = load_processed(args.base_dir / "data" / "processed")
    y = ydf["subtype"].values
    classes = sorted(pd.unique(y))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)
    yb_te = label_binarize(yte, classes=classes)

    f_vals, _ = f_classif(Xtr, ytr)
    ranked = np.argsort(np.nan_to_num(f_vals, nan=0.0))[::-1]

    rows = []
    for k in config.PANEL_SIZES:
        k = min(k, X.shape[1])
        sel = ranked[:k]
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=config.RANDOM_STATE)),
        ])
        model.fit(Xtr[:, sel], ytr)
        pred = model.predict(Xte[:, sel])
        proba = model.predict_proba(Xte[:, sel])
        proba = proba[:, [list(model.classes_).index(c) for c in classes]]
        rows.append({
            "panel_size": k,
            "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
            "test_f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
            "test_roc_auc_ovr": round(float(roc_auc_score(
                yb_te, proba, average="macro", multi_class="ovr")), 4),
        })
        r = rows[-1]
        log.info("k=%4d  acc=%.4f  f1=%.4f  auc=%.4f",
                 k, r["test_accuracy"], r["test_f1_macro"], r["test_roc_auc_ovr"])

    panel = pd.DataFrame(rows)
    panel.to_csv(results / "gene_panel.csv", index=False)

    top50 = [genes[i] for i in ranked[:50]]
    pd.DataFrame({"rank": range(1, 51), "gene": top50}).to_csv(
        results / "panel_top50_genes.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(panel["panel_size"], panel["test_accuracy"], "o-", label="accuracy")
    ax.plot(panel["panel_size"], panel["test_f1_macro"], "s-", label="macro F1")
    ax.set_xscale("log")
    ax.set_xlabel("panel size (top-k genes by ANOVA F)")
    ax.set_ylabel("test score")
    ax.set_title("PAM50 accuracy vs. gene panel size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, results / "gene_panel.png")

    log.info("wrote gene_panel.csv/png and panel_top50_genes.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
