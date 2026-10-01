"""Bits that more than one script needs."""
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on servers; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def load_processed(processed_dir: Path):
    X = np.load(processed_dir / "X.npz")["X"]
    genes = pd.read_csv(processed_dir / "genes.txt")["gene"].tolist()
    y = pd.read_csv(processed_dir / "y.csv", index_col=0)
    return X, genes, y


def save_fig(fig, path: Path, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
