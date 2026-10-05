"""Functional phase, step 1: copy the ROI time series files into your Cloud Storage bucket.

One .1D text file per subject (rows are time points, columns are brain regions), from the public
ABIDE S3 bucket, pipeline cpac, strategy filt_global, atlas cc200 by default. Nothing is saved on
your computer. Safe to rerun; files already in the bucket are skipped.

Usage:
    python src/04_download_fmri.py --probe     # test 3 subjects
    python src/04_download_fmri.py             # copy everything into the bucket
    python src/04_download_fmri.py --atlas ho  # a smaller atlas (111 regions) if you prefer
"""
import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from config import BUCKET, FMRI_ATLAS, FMRI_PIPELINE, FMRI_PREFIX, FMRI_STRATEGY, PHENO_BLOB, PROJECT

BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"


def fetch(session, file_id, url, blob_name, out_path, bucket, existing, retries=3):
    if bucket is not None:
        if blob_name in existing:
            return file_id, "cached"
    elif out_path.exists() and out_path.stat().st_size > 0:
        return file_id, "cached"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 0:
                if bucket is not None:
                    bucket.blob(blob_name).upload_from_string(r.content, content_type="text/plain")
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(r.content)
                return file_id, "ok"
            if r.status_code in (403, 404):
                return file_id, f"http_{r.status_code}"
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return file_id, "failed"


def load_pheno(bucket, local_path, project):
    if bucket is not None:
        blob = bucket.blob(PHENO_BLOB)
        if blob.exists():
            return pd.read_csv(io.StringIO(blob.download_as_text()))
    p = Path(local_path)
    if not p.exists():
        raise SystemExit(f"Phenotypic CSV not found at {p} or in the bucket. Run step 01 first.")
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pheno", default="data/raw/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--atlas", default=FMRI_ATLAS, help="cc200 (default), ho, aal, cc400, dosenbach160, ez, tt")
    ap.add_argument("--pipeline", default=FMRI_PIPELINE)
    ap.add_argument("--strategy", default=FMRI_STRATEGY)
    ap.add_argument("--local", action="store_true", help="save to local disk instead of the bucket")
    ap.add_argument("--out", default="data/raw/fmri")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    deriv = f"rois_{args.atlas}"
    prefix = f"{FMRI_PREFIX}/{args.atlas}"
    bucket, existing = None, frozenset()
    if not args.local:
        from google.cloud import storage
        client = storage.Client(project=args.project)
        try:
            bucket = client.get_bucket(args.bucket)
            existing = frozenset(b.name for b in client.list_blobs(args.bucket, prefix=prefix))
        except Exception as e:
            raise SystemExit(f"Cannot open bucket gs://{args.bucket}: {e}")
        print(f"Target: gs://{args.bucket}/{prefix}/  ({len(existing)} files already there)")

    pheno = load_pheno(bucket, args.pheno, args.project)
    ids = pheno.loc[pheno["FILE_ID"] != "no_filename", "FILE_ID"].dropna().unique().tolist()
    print(f"{len(ids)} subjects with imaging in the phenotypic file")
    if args.probe:
        ids = ids[:3]
    elif args.limit:
        ids = ids[: args.limit]

    out_dir = Path(args.out) / args.atlas
    jobs = []
    for fid in ids:
        fname = f"{fid}_{deriv}.1D"
        url = f"{BASE}/{args.pipeline}/{args.strategy}/{deriv}/{fname}"
        jobs.append((fid, url, f"{prefix}/{fname}", out_dir / fname))

    results = []
    with requests.Session() as session, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, session, fid, url, blob, path, bucket, existing)
                   for fid, url, blob, path in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if args.probe:
                fid, status = results[-1]
                print(f"{fid}: {status}")
            elif n % 100 == 0:
                print(f"  {n}/{len(jobs)} subjects processed", flush=True)

    log = pd.DataFrame(results, columns=["FILE_ID", "status"])
    print("Summary:", log["status"].value_counts().to_dict())
    bad = log[~log["status"].isin(["ok", "cached"])]
    if len(bad):
        Path("results").mkdir(exist_ok=True)
        bad.to_csv("results/fmri_download_problems.csv", index=False)
        print(f"{len(bad)} subjects had no file at this atlas/pipeline, see results/fmri_download_problems.csv")


if __name__ == "__main__":
    main()
