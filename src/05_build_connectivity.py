"""Functional phase, step 2: turn each subject's ROI time series into connectivity features.

For each subject:
    1. read the .1D time series (time points x regions) from the bucket
    2. correlate every region with every other region (Pearson)
    3. Fisher z transform the correlations (arctanh)
    4. take the upper triangle as a flat feature vector (edge_i_j)

The result is one row per subject: the connectivity edges plus the phenotypic columns
(diagnosis, age, sex, site). Because CC200 gives 19,900 edges, which exceeds BigQuery's
10,000 column limit, the table is saved as a Parquet file in your bucket, not in BigQuery.

Usage:
    python src/05_build_connectivity.py                # bucket in, Parquet in bucket out
    python src/05_build_connectivity.py --local        # local .1D files in, local Parquet out
"""
import argparse
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from config import BUCKET, FC_PARQUET, FMRI_ATLAS, FMRI_PREFIX, PHENO_BLOB, PROJECT


def read_1d(text):
    """Parse a .1D ROI time series file into a (time points x regions) float array."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line.split())
    arr = np.array(rows, dtype=float)
    return arr


def connectivity(ts):
    """Fisher z transformed upper triangle of the region by region correlation matrix."""
    # drop regions that are flat (zero variance), they produce undefined correlations
    good = ts.std(axis=0) > 0
    R = ts.shape[1]
    corr = np.full((R, R), np.nan)
    if good.sum() >= 2:
        c = np.corrcoef(ts[:, good].T)
        idx = np.where(good)[0]
        corr[np.ix_(idx, idx)] = c
    iu = np.triu_indices(R, k=1)
    edges = corr[iu]
    edges = np.clip(edges, -0.999999, 0.999999)
    z = np.arctanh(edges)
    z[np.isnan(z)] = 0.0                      # a flat region's edges carry no information
    return z, iu, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pheno", default="data/raw/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--atlas", default=FMRI_ATLAS)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out_parquet", default=FC_PARQUET, help="bucket blob name, or local path with --local")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--fmri_dir", default="data/raw/fmri", help="used only with --local")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    client = bucket = None
    if args.local:
        pheno = pd.read_csv(args.pheno)
    else:
        from google.cloud import storage
        client = storage.Client(project=args.project)
        bucket = client.bucket(args.bucket)
        pheno = pd.read_csv(io.StringIO(bucket.blob(PHENO_BLOB).download_as_text()))

    pheno = pheno[pheno["FILE_ID"] != "no_filename"].copy()
    deriv = f"rois_{args.atlas}"
    prefix = f"{FMRI_PREFIX}/{args.atlas}"
    local_dir = Path(args.fmri_dir) / args.atlas

    if args.local:
        present = [fid for fid in pheno["FILE_ID"] if (local_dir / f"{fid}_{deriv}.1D").exists()]
        read = lambda fid: (local_dir / f"{fid}_{deriv}.1D").read_text()
    else:
        names = {b.name for b in client.list_blobs(args.bucket, prefix=prefix)}
        present = [fid for fid in pheno["FILE_ID"] if f"{prefix}/{fid}_{deriv}.1D" in names]
        read = lambda fid: bucket.blob(f"{prefix}/{fid}_{deriv}.1D").download_as_text()
    print(f"{len(present)} subjects have a {args.atlas} time series file")

    def load(fid):
        try:
            ts = read_1d(read(fid))
            if ts.ndim != 2 or ts.shape[0] < 10 or ts.shape[1] < 2:
                return fid, None, None
            z, iu, R = connectivity(ts)
            return fid, z, (iu, R)
        except Exception as e:
            return fid, None, str(e)

    rows, meta, bad = {}, None, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fid, z, info in pool.map(load, present):
            if z is None:
                bad.append(fid)
            else:
                rows[fid] = z
                meta = info
    if not rows:
        raise SystemExit("No usable time series parsed. Check the atlas and that step 04 completed.")
    iu, R = meta
    edge_names = [f"edge_{i}_{j}" for i, j in zip(*iu)]
    print(f"{len(rows)} subjects usable; {R} regions; {len(edge_names)} connectivity features; {len(bad)} unusable")

    feats = pd.DataFrame.from_dict(rows, orient="index", columns=edge_names)

    fiq = pheno["FIQ"].replace(-9999, np.nan)
    base = pd.DataFrame({
        "FILE_ID": pheno["FILE_ID"].values,
        "SUB_ID": pheno["SUB_ID"].values,
        "site": pheno["SITE_ID"].values,
        "site_group": pheno["SITE_ID"].str.replace(r"_\d+$", "", regex=True).values,
        "dx": (pheno["DX_GROUP"] == 1).astype(int).values,
        "age": pheno["AGE_AT_SCAN"].values,
        "sex_female": (pheno["SEX"] == 2).astype(int).values,
        "fiq": fiq.values,
    }).set_index("FILE_ID")

    df = base.join(feats, how="inner").reset_index().rename(columns={"index": "FILE_ID"})
    print(f"FINAL connectivity table: {len(df)} subjects x {df.shape[1]} columns "
          f"(autism {int(df.dx.sum())}, control {int((1 - df.dx).sum())}; sites {df.site_group.nunique()})")

    if args.local:
        Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out_parquet, index=False)
        print(f"Wrote {args.out_parquet}")
    else:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        bucket.blob(args.out_parquet).upload_from_string(buf.getvalue(),
                                                         content_type="application/octet-stream")
        print(f"Wrote gs://{args.bucket}/{args.out_parquet} ({buf.tell() / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
