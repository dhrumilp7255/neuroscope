"""One place for the cloud names. Change them here if your project ID or bucket name is different."""
PROJECT = "neuroscope26"                                   # GCP project ID
BUCKET = "neuroscope26_data"                               # Cloud Storage bucket
FS_PREFIX = "abide/freesurfer"                             # stats files live at gs://BUCKET/FS_PREFIX/FILE_ID/*.stats
PHENO_BLOB = "abide/phenotypic/Phenotypic_V1_0b_preprocessed1.csv"
BQ_TABLE = "neuroscope26.abide.features"                   # project.dataset.table

# Functional connectivity phase
FMRI_PIPELINE = "cpac"
FMRI_STRATEGY = "filt_global"
FMRI_ATLAS = "cc200"                                       # rois_cc200 (200 regions, 19,900 connectivity features)
FMRI_PREFIX = "abide/fmri"                                 # time series land at gs://BUCKET/FMRI_PREFIX/{atlas}/FILE_ID_rois_{atlas}.1D
FC_PARQUET = "abide/fmri/connectivity_cc200.parquet"       # connectivity feature table (Parquet in the bucket, not BigQuery)
