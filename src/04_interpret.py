#!/usr/bin/env python3
"""Figure out what the models actually learned — and where they mess up.

Three things:
  1. Top genes per model (|coef| for logreg, importances for the trees).
  2. Sanity check: do the usual breast-cancer suspects (ESR1, ERBB2, MKI67,
     ...) show up? If the top genes were all random, something would be off.
  3. Misclassification analysis: normalize the confusion matrix by row and
     look at which subtype pairs get mixed up. Biology says LumA vs LumB
     should be the messy boundary (both ER+, split mostly by proliferation),
     so that's the hypothesis going in.
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
from sklearn.model_selection import train_test_split

import config
from utils import load_processed, save_fig, setup_logging

log = logging.getLogger(__name__)

# not a discovery list — just markers we'd expect to see if things are sane
MARKERS = {"ESR1", "ERBB2", "MKI67", "FOXA1", "GATA3", "PGR", "KRT5", "KRT14",
           "EGFR", "CCND1", "MYC", "TP53", "FOXC1", "MLPH", "BCL2"}


def make_models():
    # same three models as 03; copied here so this script runs on its own
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logreg": Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000,
                                                       class_weight="balanced",
                                                       random_state=config.RANDOM_STATE))]),
        "random_forest": Pipeline([("scaler", StandardScaler()),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=300, n_jobs=-1,
                                       class_weight="balanced_subsample",
                                       random_state=config.RANDOM_STATE))]),
        "hist_gb": Pipeline([("scaler", StandardScaler()),
                             # same as 03: no class_weight, it's 4-5x slower with it
                             ("clf", HistGradientBoostingClassifier(
                                 random_state=config.RANDOM_STATE))]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=config.BASE_DIR)
    args = p.parse_args()
    setup_logging()

    results = args.base_dir / "results"
    X, genes, ydf = load_processed(args.base_dir / "data" / "processed")
    y = ydf["subtype"].values
    classes = sorted(pd.unique(y))
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)

    rankings = {}
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            score = np.abs(clf.coef_).max(axis=0)  # strongest class signal wins
            how = "max |coef|"
        elif hasattr(clf, "feature_importances_"):
            score = clf.feature_importances_
            how = "importance"
        else:
            # gradient boosting doesn't expose importances, so permute instead.
            # refit on just the 500 highest-variance genes first — permuting
            # all 3000 three times would take longer than the training itself.
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.inspection import permutation_importance
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            cand = np.argsort(Xtr.var(axis=0))[::-1][:500]
            small = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", HistGradientBoostingClassifier(
                    random_state=config.RANDOM_STATE)),
            ])
            small.fit(Xtr[:, cand], ytr)
            perm = permutation_importance(small, Xtr[:, cand], ytr, n_repeats=3,
                                          random_state=config.RANDOM_STATE,
                                          n_jobs=1)
            score = np.zeros(Xtr.shape[1])
            score[cand] = perm.importances_mean
            how = "permutation importance (top-500 variance genes)"
        top = np.argsort(score)[::-1][: config.TOP_K]
        rankings[name] = [(genes[i], float(score[i])) for i in top]
        log.info("%s top genes (%s): %s", name, how,
                 [g for g, _ in rankings[name][:8]])

    rows = [{"model": m, "rank": r + 1, "gene": g, "score": round(s, 4)}
            for m, ranked in rankings.items()
            for r, (g, s) in enumerate(ranked)]
    pd.DataFrame(rows).to_csv(results / "top_genes.csv", index=False)

    marker_hits = {}
    for name, ranked in rankings.items():
        hits = sorted({g for g, _ in ranked} & MARKERS)
        marker_hits[name] = hits
        log.info("known markers in %s top-%d: %s", name, config.TOP_K,
                 hits if hits else "none — worth a second look")

    # --- misclassifications from the best model's test predictions ---
    metrics = json.loads((results / "metrics.json").read_text())
    best = metrics["best_model"]
    cm = np.array(metrics["models"][best]["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)

    confusions = []
    for i, true_c in enumerate(classes):
        for j, pred_c in enumerate(classes):
            if i != j and cmn[i, j] >= 0.05:
                confusions.append({"true": true_c, "predicted_as": pred_c,
                                   "fraction": round(float(cmn[i, j]), 3),
                                   "n": int(cm[i, j])})
    confusions.sort(key=lambda d: -d["fraction"])
    for c in confusions:
        log.info("mix-up: %s -> %s (%.0f%% of true %s)",
                 c["true"], c["predicted_as"], 100 * c["fraction"], c["true"])

    metrics["interpretation"] = {
        "known_markers_in_top_genes": marker_hits,
        "per_class_recall": {c: round(float(cmn[i, i]), 3)
                             for i, c in enumerate(classes)},
        "notable_confusions": confusions,
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lr_top = rankings["logreg"][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([g for g, _ in lr_top][::-1], [s for _, s in lr_top][::-1])
    ax.set_xlabel("max |coefficient| across classes")
    ax.set_title("top 15 genes — logistic regression")
    save_fig(fig, results / "top_genes.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"confusion matrix, row-normalized — {best}")
    save_fig(fig, results / "confusion_matrix_normalized.png")

    log.info("wrote top_genes.csv/png and confusion_matrix_normalized.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
