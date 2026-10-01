#!/usr/bin/env python3
"""Download the TCGA-BRCA data we need.

I originally wanted this from UCSC Xena's GDC hub, but their S3 download
links started returning 403 AccessDenied (checked Sep 2026, even with a
browser user-agent). So instead this pulls the same TCGA-BRCA cohort from
cBioPortal's public datahub, which serves its Git-LFS files over plain HTTPS:

  - brca_tcga: RNA-Seq V2 RSEM matrix + clinical files
  - brca_tcga_pub (the 2012 Nature paper cohort): clinical file with the
    PAM50_SUBTYPE calls we classify

Everything goes to data/raw/, which is gitignored — rerun this script to
get the data back.
"""
import argparse
import logging
import sys
from pathlib import Path

import requests

import config
from utils import setup_logging

log = logging.getLogger(__name__)

LFS = "https://media.githubusercontent.com/media/cBioPortal/datahub/master"
FILES = [
    ("public/brca_tcga/data_mrna_seq_v2_rsem.txt", "data_mrna_seq_v2_rsem.txt"),  # ~192 MB
    ("public/brca_tcga/data_clinical_sample.txt", "data_clinical_sample.txt"),
    ("public/brca_tcga/data_clinical_patient.txt", "data_clinical_patient.txt"),
    ("public/brca_tcga_pub/data_clinical_sample.txt", "brca_tcga_pub_clinical_sample.txt"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=config.BASE_DIR)
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()
    setup_logging()

    raw = args.base_dir / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    for repo_path, local_name in FILES:
        dest = raw / local_name
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            log.info("skip %s (already there)", local_name)
            continue
        log.info("fetching %s ...", local_name)
        with requests.get(f"{LFS}/{repo_path}", stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        log.info("saved %s (%.1f MB)", local_name, dest.stat().st_size / 1e6)

    log.info("done — next: python3 src/02_preprocess.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
