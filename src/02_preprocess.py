#!/usr/bin/env python3
"""Turn the raw downloads into a labeled feature matrix.

Labels come from the 2012 TCGA paper cohort's PAM50_SUBTYPE column
(LumA / LumB / Basal / HER2). Normal-like only has 8 samples, so it gets
dropped — not enough to learn from or evaluate on.

The matrix is saved unscaled on purpose: 03 fits the scaler inside each CV
fold, so there's no leakage from the test set into training.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import setup_logging

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=config.BASE_DIR)
    args = p.parse_args()
    setup_logging()

    raw = args.base_dir / "data" / "raw"
    proc = args.base_dir / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    # --- expression ---
    expr = pd.read_csv(raw / "data_mrna_seq_v2_rsem.txt", sep="\t")
    expr = expr.drop(columns=["Entrez_Gene_Id"])
    # a few genes appear twice; median is safer than mean here
    expr = expr.groupby("Hugo_Symbol", as_index=False).median(numeric_only=True)
    X = expr.set_index("Hugo_Symbol").T
    X.index.name = "sample_id"
    log.info("expression: %d samples x %d genes", *X.shape)

    # --- labels ---
    clin = pd.read_csv(raw / "brca_tcga_pub_clinical_sample.txt",
                       sep="\t", comment="#", low_memory=False)
    clin = clin.dropna(subset=["PAM50_SUBTYPE"])
    clin["subtype"] = clin["PAM50_SUBTYPE"].map(config.SUBTYPE_MAP)
    counts = clin["subtype"].value_counts()
    keep = counts[counts >= config.MIN_CLASS_SAMPLES].index.tolist()
    log.info("dropping %s (<%d samples each)",
             sorted(set(counts.index) - set(keep)), config.MIN_CLASS_SAMPLES)
    y = clin[clin["subtype"].isin(keep)].set_index("SAMPLE_ID")["subtype"]
    log.info("label counts:\n%s", y.value_counts())

    # --- align, transform, filter ---
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    log.info("%d samples with both expression and label", len(common))

    X = np.log2(X + 1)  # RSEM counts are linear-scale
    top = X.var(axis=0).sort_values(ascending=False).head(config.N_TOP_GENES).index
    X = X[top].astype(np.float32)
    log.info("kept top %d genes by variance -> %s", config.N_TOP_GENES, X.shape)

    np.savez_compressed(proc / "X.npz", X=X.values)
    X.index.to_series().to_csv(proc / "samples.txt", index=False, header=False)
    pd.Series(X.columns, name="gene").to_csv(proc / "genes.txt", index=False)
    y.loc[X.index].to_frame("subtype").to_csv(proc / "y.csv")

    meta = {
        "n_samples": X.shape[0],
        "n_genes": X.shape[1],
        "class_counts": y.value_counts().to_dict(),
        "transform": "log2(RSEM+1)",
        "label_source": "cBioPortal brca_tcga_pub PAM50_SUBTYPE (TCGA 2012 paper)",
        "dropped": "Normal-like, n=8",
        "random_state": config.RANDOM_STATE,
    }
    (proc / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote data/processed/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
