#  Knee_MRI_Segmentation_2.5D
<p align="justify">This GitHub repository includes all Python codes, annotation guideline, fully annotated imaging dataset, AI models, and a lightweight software applicationt to offer a clinically oriented and explainable deep learning framework, namely <strong>KneeXNet-2.5D</strong>, for accurate and efficient knee cartilage and meniscus segmentation in sagittal MRIs. Unlike traditional 3D segmentation methods, the proposed model employs a 2.5D architecture to capture the inter-slice spatial context, achieving high segmentation accuracy while maintaining computational efficiency and optimal resource utilization. To enable open scientific research and ensure reproducibility, we have made this GitHub repository publicly and freely available for any research and educational purposes. 
</p>

### Directory Descriptions:
+ <p align="justify"><strong>Code:</strong> This directory contains all the Python code we implemented to carry out the study.</p>
+ <p align="justify"><strong>Annotation_guideline:</strong> This directory contains the annotation guidelines developed by domain experts for creating manual segmentation masks. </p>
+ <p align="justify"><strong>Fully_annotated_dataset:</strong> This directory contains the fully annotated imaging dataset used in this study, featuring the gold-standard manual segmentations.</p>
+ <p align="justify"><strong>Lightweight_application:</strong> This directory includes a lightweight, interactive software application designed to deploy KneeXNet-2.5D in both clinical and research settings.</p>
+ <p align="justify"><strong>Models:</strong> This directory includes all AI models developed for this study.</p>






### The proposed computational pipeline:

![alt text](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D/blob/main/Figures/pipeline.png  "The proposed computational pipeline")
</p>
<p>
</p>

### Osteoarthritis Initiative (OAI) dataset: 
<p>The authors thank the <a href="https://nda.nih.gov/oai" target="_blank"> Osteoarthritis Initiative (OAI)</a> for the datasets utilized in this research contribution.</p>

### Acknowledgements:
<p align="justify"> We gratefully acknowledge the Department of Health Information Management at the School of Health and Rehabilitation Sciences (SHRS), University of Pittsburgh, for providing the computational infrastructure and support that enabled the training and evaluation of our deep learning models. Their resources and technical assistance were instrumental in the successful completion of this study. </p>

### Citation:

<p align="justify">This contribution is fully described in a manuscript currently under review at <a href="https://www.nature.com/npjhealthsyst/" target="_blank">npj Health Systems</a>. 

  ```
  @article{siddiqui2024fair,
  title={Fair AI-powered orthopedic image segmentation: addressing bias and promoting equitable healthcare},
  author={Siddiqui, Ismaeel A and Littlefield, Nickolas and Carlson, Luke A and Gong, Matthew and Chhabra, Avani and Menezes, Zoe and Mastorakos, George M and Thakar, Sakshi Mehul and Abedian, Mehrnaz and Lohse, Ines and others},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={16105},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```





















# knee-cartilage-segmentation
## Overview
This repository presents a deep learning pipeline for automated knee cartilage segmentation using T2-weighted sagittal MRI scans. The project is motivated by the clinical need for accurate, scalable tools to assess cartilage degradation, a key indicator of osteoarthritis, one of the most prevalent musculoskeletal conditions worldwide. Early detection and monitoring of cartilage degradation are essential for timely intervention and improved patient outcomes.

The workflow integrates a YOLOv11-based localization model with a 2.5D U-Net segmentation architecture to efficiently identify cartilage regions while preserving anatomical context across adjacent slices. By leveraging advanced computational techniques, this project offers an innovative and reproducible solution to the challenges of musculoskeletal imaging, addressing critical issues of time, scalability, and consistency in both research and clinical settings.

## Dataset
This project uses T2-weighted sagittal MRI scans from the publicly available Osteoarthritis Initiative (OAI) dataset, which provides longitudinal imaging data for assessing the progression of knee osteoarthritis. A subset of MRIs was manually annotated to generate segmentation masks for key cartilage structures, including the femoral, tibial, patellar, and meniscal regions. The dataset was split into training, validation, and test sets, with the final evaluation conducted on held-out, unseen data to ensure unbiased performance assessment.

Instructions for accessing the original dataset are available through the [[OAI website](https://nda.nih.gov/oai)].

## Evaluation metrics
Model performance was assessed using:
* Mean Average Precision (mAP) to evaluate the localization accuracy of the YOLOv11 detection.
* Intersection over Union (IoU) and Dice Similarity Coefficient (DSC) to assess segmentation accuracy.
