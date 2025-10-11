## 2.5D MRI Segmentation Annotations

This directory provides manually annotated knee cartilage segmentation masks for the Osteoarthritis Initiative (OAI) dataset. These annotations support research in knee cartilage segmentation, joint space width measurement, and 2.5D deep learning. While the masks are included here, users must individually download the corresponding MRI or X-ray data from the official OAI repository.

## Annotation Structure
Each annotation file follows the format:

```<PatientID>_<DateOfMRI>_<MRICode>.nii.gz```

with:
- Patient ID: the Unique OAI participant identifier.

- Date of MRI: Date the MRI was acquired (YYYYMMDD format).

- MRI Code: Unique imaging barcode corresponding to a specific MRI series.

All masks are stored in NIFTI (.nii.gz) format and aligned with the sagittal MRIs from the OAI dataset.

## Downloading The Original Dataset
To use these annotations, you must first download the corresponding MRIs from the official OAI repository.

Steps:
1. Visit the OAI Portal: https://nda.nih.gov/oai
2. Create or log in to your NIH Data Archive (NDA) account.
3. Request access to the OAI dataset and agree to its data use terms.
4. Once approved, download baseline (00m) the knee MRI scans for the desired subjects.
5. Match the MRI scans to the annotations using the filename convention above.

## Usage
Load the NIfTI annotation files using nibabel or any compatible medical imaging library:

```python
import nibabel as nib
mask = nib.load("9001104_20050825_10498205.nii.gz").get_fdata()
```

Pair with the corresponding MRI from OAI.
