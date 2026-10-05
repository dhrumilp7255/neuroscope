"""Phase 1: copy FreeSurfer stats files for every ABIDE I subject straight into your Cloud Storage bucket.

Nothing is saved on your computer. Each file is downloaded into memory from the public S3 bucket and uploaded
to gs://BUCKET/abide/freesurfer/FILE_ID/. The phenotypic CSV is uploaded to the bucket as well.

Per subject we copy:
    stats/aseg.stats      subcortical, cerebellar and ventricle volumes, total intracranial volume
    stats/lh.aparc.stats  left hemisphere cortical thickness, surface area, gray matter volume
    stats/rh.aparc.stats  right hemisphere, same measures

Usage:
    python src/01_download_freesurfer.py --probe     # test 3 subjects first
    python src/01_download_freesurfer.py             # copy everything into the bucket (safe to rerun, skips files already there)
    python src/01_download_freesurfer.py --local     # only if you really want files on your own disk
"""
import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from config import BUCKET, FS_PREFIX, PHENO_BLOB, PROJECT

BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1"
FILES = ["aseg.stats", "lh.aparc.stats", "rh.aparc.stats"]


def fetch(session, file_id, name, out_dir, bucket=None, prefix=None, existing=frozenset(), retries=3):
    blob_name = f"{prefix}/{file_id}/{name}"
    dest = out_dir / file_id / name
    if bucket is not None:
        if blob_name in existing:
            return file_id, name, "cached"
    elif dest.exists() and dest.stat().st_size > 0:
        return file_id, name, "cached"
    url = f"{BASE}/{file_id}/stats/{name}"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                if bucket is not None:
                    bucket.blob(blob_name).upload_from_string(r.content, content_type="text/plain")
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(r.content)
                return file_id, name, "ok"
            if r.status_code in (403, 404):
                return file_id, name, f"http_{r.status_code}"
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return file_id, name, "failed"


def load_pheno(bucket, local_path):
    """Use the copy in the bucket if there is one, otherwise upload the local CSV from the repo to the bucket."""
    if bucket is not None:
        blob = bucket.blob(PHENO_BLOB)
        if blob.exists():
            return pd.read_csv(io.StringIO(blob.download_as_text()))
    p = Path(local_path)
    if not p.exists():
        raise SystemExit(f"Phenotypic CSV not found at {p}. Put it there, or upload it to gs://{BUCKET}/{PHENO_BLOB}.")
    if bucket is not None:
        bucket.blob(PHENO_BLOB).upload_from_filename(str(p))
        print(f"Uploaded phenotypic CSV to gs://{BUCKET}/{PHENO_BLOB}")
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pheno", default="data/raw/Phenotypic_V1_0b_preprocessed1.csv", help="local copy, uploaded once")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", default=FS_PREFIX)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--local", action="store_true", help="save to local disk instead of the bucket")
    ap.add_argument("--out", default="data/raw/freesurfer", help="used only with --local")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N subjects")
    ap.add_argument("--probe", action="store_true", help="try 3 subjects and print the URLs and status codes")
    args = ap.parse_args()

    bucket, existing = None, frozenset()
    if not args.local:
        from google.cloud import storage
        client = storage.Client(project=args.project)
        try:
            bucket = client.get_bucket(args.bucket)
            existing = frozenset(b.name for b in client.list_blobs(args.bucket, prefix=args.prefix))
        except Exception as e:
            raise SystemExit(f"Cannot open bucket gs://{args.bucket}: {e}\n"
                             f"Create it first: gcloud storage buckets create gs://{args.bucket} --location=us-east1")
        print(f"Target: gs://{args.bucket}/{args.prefix}/  ({len(existing)} files already there)")

    pheno = load_pheno(bucket, args.pheno)
    ids = pheno.loc[pheno["FILE_ID"] != "no_filename", "FILE_ID"].dropna().unique().tolist()
    print(f"{len(ids)} subjects with imaging in the phenotypic file")
    if args.probe:
        ids = ids[:3]
    elif args.limit:
        ids = ids[: args.limit]

    out_dir = Path(args.out)
    jobs = [(i, f) for i in ids for f in FILES]
    results = []
    with requests.Session() as session, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, session, i, f, out_dir, bucket, args.prefix, existing) for i, f in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if args.probe:
                fid, name, status = results[-1]
                print(f"{BASE}/{fid}/stats/{name}  ->  {status}")
            elif n % 300 == 0:
                print(f"  {n}/{len(jobs)} files processed", flush=True)

    log = pd.DataFrame(results, columns=["FILE_ID", "file", "status"])
    print("Summary:", log["status"].value_counts().to_dict())
    bad = log[~log["status"].isin(["ok", "cached"])]
    if len(bad):
        Path("results").mkdir(exist_ok=True)
        bad.to_csv("results/download_problems.csv", index=False)
        print(f"{len(bad)} files had problems, see results/download_problems.csv")


if __name__ == "__main__":
    main()