# Data Access

This project was developed with MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).

Raw MRI scans, extracted slices, subject identifiers, HDF5 files, and trained checkpoints are not redistributed in this repository because ADNI data is governed by a data use agreement.

To reproduce the training pipeline, obtain authorized ADNI access and prepare an HDF5 file with these keys:

| Key | Description | Expected shape |
|---|---|---|
| `images` | Preprocessed 2D coronal MRI slices | `(N, C, 224, 224)` |
| `labels` | Binary class labels, `0 = CN`, `1 = AD` | `(N,)` |
| `subject_ids` | Subject-level grouping identifiers for cross-validation | `(N,)` |

The training script expects the HDF5 file at `data/adcn_slices.h5` by default. You can also pass a path directly:

```bash
python src/train_subject_cv.py --hdf5-path /path/to/adcn_slices.h5
```

Do not commit ADNI-derived data files to GitHub.
