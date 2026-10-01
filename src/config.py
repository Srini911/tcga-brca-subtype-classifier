"""Config for the BRCA PAM50 project.

All the magic numbers in one place so the scripts stay readable.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 5
N_TOP_GENES = 3000      # variance filter in 02_preprocess
MIN_CLASS_SAMPLES = 10  # Normal-like has n=8, so it gets dropped in 02
PANEL_SIZES = (3000, 500, 100, 50, 20, 10)  # 05_gene_panel
TOP_K = 20              # interpretation lists in 04

# Short labels used everywhere downstream.
SUBTYPE_MAP = {
    "Luminal A": "LumA",
    "Luminal B": "LumB",
    "Basal-like": "Basal",
    "HER2-enriched": "HER2",
    "Normal-like": "Normal",
}
