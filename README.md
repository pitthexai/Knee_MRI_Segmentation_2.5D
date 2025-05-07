# knee-cartilage-segmentation
## Overview
This project presents a deep learning pipeline for automated knee cartilage segmentation using T2-weighted sagittal MRI scans. The project is motivated by the clinical need for accurate, scalable tools to assess cartilage degradation, a key indicator of osteoarthritis, one of the most prevalent musculoskeletal conditions worldwide. Early detection and monitoring of cartilage degradation are essential for timely intervention and improved patient outcomes.

The workflow integrates a YOLOv11-based localization model with a 2.5D U-Net segmentation architecture to efficiently identify cartilage regions while preserving anatomical context across adjacent slices. By leveraging advanced computational techniques, this project offers an innovative and reproducible solution to the challenges of musculoskeletal imaging, addressing critical issues of time, scalability, and consistency in both research and clinical settings.

## Dataset
## Evaluation metrics
Model performance was assessed using:
* Mean Average Precision (mAP) to evaluate the localization accuracy of the YOLOv11 detection.
* Intersection over Union (IoU) and Dice Similarity Coefficient (DSC) to assess segmentation accuracy.
