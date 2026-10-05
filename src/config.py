"""One place for the cloud names. Change them here if your project ID or bucket name is different."""
PROJECT = "<Project_Name>"                                   # GCP project ID
BUCKET = "<Bucket_Name>"                               # Cloud Storage bucket
FS_PREFIX = "<PREFIX>"                             # stats files live at gs://BUCKET/FS_PREFIX/FILE_ID/*.stats
PHENO_BLOB = "<YOUR_PATH>"
BQ_TABLE = "<TABLE>"                   # project.dataset.table

# Functional connectivity phase
FMRI_PIPELINE = "<__>"
FMRI_STRATEGY = "<__>"
FMRI_ATLAS = "<__>"                                       # rois_cc200 (200 regions, 19,900 connectivity features)
FMRI_PREFIX = "<__>"                                 # time series land at gs://BUCKET/FMRI_PREFIX/{atlas}/FILE_ID_rois_{atlas}.1D
FC_PARQUET = "<__>"       # connectivity feature table (Parquet in the bucket, not BigQuery)
