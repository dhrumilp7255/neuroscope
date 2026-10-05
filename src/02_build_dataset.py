"""Phases 2 and 3: parse FreeSurfer stats files, merge with phenotypic data, clean, QC, engineer features.

Output: data/processed/dataset.csv (one row per participant) and results/dataset_report.txt (row counts at each step).

Default (cloud): reads the stats files and the phenotypic CSV from your Cloud Storage bucket, processes them in
memory on your PC, and writes the final table to BigQuery. Nothing is stored on your computer.
    python src/02_build_dataset.py
Names come from src/config.py. Add --no-qc to keep subjects that failed anatomical QC.
Add --local to read data/raw/freesurfer and write data/processed/dataset.csv instead.
"""
import argparse
import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from config import BQ_TABLE, BUCKET, FS_PREFIX, PHENO_BLOB, PROJECT

ICV_NAMES = ["eTIV", "ICV", "EstimatedTotalIntraCranialVol"]
APARC_GLOBALS = ["MeanThickness", "WhiteSurfArea"]


def clean(name):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")


FILES = ("aseg.stats", "lh.aparc.stats", "rh.aparc.stats")


def parse_stats(text, label="stats file"):
    """Return (measures dict, table DataFrame) from the text of a FreeSurfer .stats file."""
    measures, cols, rows = {}, None, []
    for line in text.splitlines():
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line[len("# Measure"):].split(",")]
            if len(parts) >= 4:
                try:
                    measures[parts[1]] = float(parts[3])
                except ValueError:
                    pass
        elif line.startswith("# ColHeaders"):
            cols = line.split()[2:]
        elif line.strip() and not line.startswith("#"):
            rows.append(line.split())
    if not cols or not rows:
        raise ValueError(f"no table found in {label}")
    table = pd.DataFrame(rows, columns=cols)
    for c in table.columns:
        if c != "StructName":
            table[c] = pd.to_numeric(table[c], errors="coerce")
    return measures, table


def subject_features(read):
    """Raw regional features for one subject. read(filename) returns that file's text."""
    feats = {}
    m, t = parse_stats(read("aseg.stats"), "aseg.stats")
    for k, v in m.items():
        feats[f"global_{clean(k)}"] = v
    for _, r in t.iterrows():
        feats[f"vol_{clean(r['StructName'])}"] = r["Volume_mm3"]
    for hemi in ("lh", "rh"):
        m, t = parse_stats(read(f"{hemi}.aparc.stats"), f"{hemi}.aparc.stats")
        for k in APARC_GLOBALS:
            if k in m:
                feats[f"global_{hemi}_{k}"] = m[k]
        for _, r in t.iterrows():
            region = clean(r["StructName"])
            feats[f"thk_{hemi}_{region}"] = r["ThickAvg"]
            feats[f"area_{hemi}_{region}"] = r["SurfArea"]
            feats[f"gvol_{hemi}_{region}"] = r["GrayVol"]
    return feats


def engineer(raw):
    """Head size normalization and left right asymmetry indices."""
    icv_col = next((f"global_{n}" for n in ICV_NAMES if f"global_{n}" in raw.columns), None)
    if icv_col is None:
        raise ValueError("no intracranial volume measure found in aseg.stats")
    icv = raw[icv_col]
    cols = {"icv": icv}
    for c in raw.columns:
        if c == icv_col:
            continue
        if c.startswith("thk_") or c.endswith("MeanThickness"):
            cols[c] = raw[c]                                  # thickness stays in mm
        elif c.startswith("area_") or c.endswith("WhiteSurfArea"):
            cols[f"rel_{c}"] = raw[c] / icv ** (2 / 3)        # areas scale roughly with ICV^(2/3)
        elif c.startswith(("vol_", "gvol_")) or (c.startswith("global_") and c.endswith("Vol")):
            cols[f"rel_{c}"] = raw[c] / icv                   # volumes as a fraction of head size
        # other aseg header numbers are not used
    # asymmetry index (L - R) / (L + R) for every left right pair
    base = dict(cols)
    for c in list(base):
        for left, right in (("_lh_", "_rh_"), ("_Left_", "_Right_")):
            if left in c:
                partner = c.replace(left, right)
                if partner in base:
                    denom = base[c] + base[partner]
                    cols[f"asym_{c.replace(left, '_')}"] = (base[c] - base[partner]) / denom.replace(0, np.nan)
    return pd.DataFrame(cols, index=raw.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="use local files and a local CSV instead of the cloud")
    ap.add_argument("--pheno", default="data/raw/Phenotypic_V1_0b_preprocessed1.csv", help="used only with --local")
    ap.add_argument("--fs_dir", default="data/raw/freesurfer", help="used only with --local")
    ap.add_argument("--out", default="data/processed/dataset.csv", help="used only with --local")
    ap.add_argument("--report", default="results/dataset_report.txt")
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", default=FS_PREFIX)
    ap.add_argument("--bq_table", default=BQ_TABLE)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_missing", type=float, default=0.2, help="drop subjects missing more than this share of features")
    args = ap.parse_args()

    log = []
    client = bucket = None
    if args.local:
        pheno = pd.read_csv(args.pheno)
    else:
        from google.cloud import storage
        client = storage.Client(project=args.project)
        bucket = client.bucket(args.bucket)
        pheno = pd.read_csv(io.StringIO(bucket.blob(PHENO_BLOB).download_as_text()))
    log.append(f"Rows in phenotypic file: {len(pheno)}")
    pheno = pheno[pheno["FILE_ID"] != "no_filename"].copy()
    log.append(f"Rows with an imaging FILE_ID: {len(pheno)}")

    # phenotypic side: only the columns we intend to use (no ADOS, ADI R, SRS, medication, comorbidity)
    fiq = pheno["FIQ"].replace(-9999, np.nan)
    base = pd.DataFrame({
        "FILE_ID": pheno["FILE_ID"],
        "SUB_ID": pheno["SUB_ID"],
        "site": pheno["SITE_ID"],
        "site_group": pheno["SITE_ID"].str.replace(r"_\d+$", "", regex=True),
        "dx": (pheno["DX_GROUP"] == 1).astype(int),         # 1 = autism, 0 = control
        "age": pheno["AGE_AT_SCAN"],
        "sex_female": (pheno["SEX"] == 2).astype(int),
        "fiq": fiq,
        "qc_fail": (pheno["qc_anat_rater_2"] == "fail") | (pheno["qc_anat_rater_3"] == "fail"),
    })

    # imaging side: read each subject's three files from Cloud Storage or from local disk
    fs_dir = Path(args.fs_dir)
    if not args.local:
        names = {b.name for b in client.list_blobs(args.bucket, prefix=args.prefix)}
        has_all = lambda fid: all(f"{args.prefix}/{fid}/{f}" in names for f in FILES)
        make_reader = lambda fid: (lambda f: bucket.blob(f"{args.prefix}/{fid}/{f}").download_as_text())
    else:
        has_all = lambda fid: all((fs_dir / fid / f).exists() for f in FILES)
        make_reader = lambda fid: (lambda f: (fs_dir / fid / f).read_text())

    present = [fid for fid in base["FILE_ID"] if has_all(fid)]
    missing = [fid for fid in base["FILE_ID"] if fid not in set(present)]

    def load(fid):
        try:
            return fid, subject_features(make_reader(fid)), None
        except Exception as e:  # unreadable or truncated file
            return fid, None, str(e)

    rows, broken = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fid, feats_i, err in pool.map(load, present):
            if err:
                broken.append((fid, err))
            else:
                rows[fid] = feats_i
    raw = pd.DataFrame.from_dict(rows, orient="index")
    log.append(f"Subjects with all three stats files parsed: {len(raw)}")
    log.append(f"Subjects without stats files: {len(missing)}; unreadable files: {len(broken)}")

    feats = engineer(raw)
    df = base.merge(feats, left_on="FILE_ID", right_index=True, how="inner")
    log.append(f"Rows after merge on FILE_ID: {len(df)}")

    if not args.no_qc:
        n = len(df)
        df = df[~df["qc_fail"]]
        log.append(f"Dropped for failed anatomical QC (rater 2 or 3): {n - len(df)}")
    df = df.drop(columns="qc_fail")

    feat_cols = [c for c in df.columns if c not in
                 ["FILE_ID", "SUB_ID", "site", "site_group", "dx", "age", "sex_female", "fiq"]]
    miss_share = df[feat_cols].isna().mean(axis=1)
    n = len(df)
    df = df[miss_share <= args.max_missing]
    log.append(f"Dropped for more than {args.max_missing:.0%} missing features: {n - len(df)}")

    const = [c for c in feat_cols if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=const)
    feat_cols = [c for c in feat_cols if c not in const]
    log.append(f"Dropped {len(const)} constant or empty feature columns")

    log.append(f"FINAL subjects: {len(df)} (autism {int(df.dx.sum())}, control {int((1 - df.dx).sum())}); "
               f"with FIQ: {int(df.fiq.notna().sum())}; sites: {df.site_group.nunique()}")
    log.append(f"FINAL MRI features: {len(feat_cols)} "
               f"(thickness {sum(c.startswith('thk_') for c in feat_cols)}, "
               f"relative area {sum(c.startswith('rel_area') for c in feat_cols)}, "
               f"relative volume {sum(c.startswith('rel_') and 'vol' in c for c in feat_cols)}, "
               f"asymmetry {sum(c.startswith('asym_') for c in feat_cols)})")

    if not args.local:
        from google.cloud import bigquery
        bq = bigquery.Client(project=args.project)
        job = bq.load_table_from_dataframe(
            df, args.bq_table, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
        job.result()
        log.append(f"Wrote {len(df)} rows x {df.shape[1]} columns to BigQuery table {args.bq_table}")
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(log) + "\n")
    print("\n".join(log))
    if missing:
        print(f"First missing FILE_IDs: {missing[:5]}")


if __name__ == "__main__":
    main()