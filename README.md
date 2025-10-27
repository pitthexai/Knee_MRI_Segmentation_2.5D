#  Knee_MRI_Segmentation_2.5D

## Table of Contents
- [Abstract](#abstract)
- [Directory Descriptions](#directory-descriptions)
- [The Proposed Computational Pipeline](#the-proposed-computational-pipeline)
- [KneeXNet-2.5D: App Interface](#kneexnet-25d-app-interface)
- [The Osteoarthritis Initiative (OAI) Dataset](#the-osteoarthritis-initiative-oai-dataset)
- [Publications](#publications)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)





### Abstract
<p align="justify">This GitHub repository includes all Python codes, annotation guideline, fully annotated imaging dataset, AI models, and a lightweight software applicationt to offer a clinically oriented and explainable deep learning framework, namely <strong>KneeXNet-2.5D</strong>, for accurate and efficient knee cartilage and meniscus segmentation in sagittal MRIs. Unlike traditional 3D segmentation methods, the proposed model employs a 2.5D architecture to capture the inter-slice spatial context, achieving high segmentation accuracy while maintaining computational efficiency and optimal resource utilization. To enable open scientific research and ensure reproducibility, we have made this GitHub repository publicly and freely available for any research and educational purposes. 
</p>



### Directory Descriptions
+ <p align="justify"><strong>Code:</strong> This directory contains all the Python code we implemented to carry out the study.</p>
+ <p align="justify"><strong>Annotation_guideline:</strong> This directory contains the annotation guidelines developed by domain experts for creating manual segmentation masks. </p>
+ <p align="justify"><strong>Fully_annotated_dataset:</strong> This directory contains the fully annotated imaging dataset used in this study, featuring the gold-standard manual segmentations.</p>
+ <p align="justify"><strong>Lightweight_application:</strong> This directory includes a lightweight, interactive software application designed to deploy KneeXNet-2.5D in both clinical and research settings.</p>
+ <p align="justify"><strong>Models:</strong> This directory includes all AI models developed for this study.</p>






### The proposed computational pipeline

![alt text](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D/blob/main/Figures/pipeline.png  "The proposed computational pipeline")
</p>
<p>
</p>

### KneeXNet-2.5D: App Interface 

![alt text](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D/blob/main/Figures/app.png  "The software app")
<p align="center">
  <a href="https://www.youtube.com/watch?v=rZD0lrhb_KE" target="_blank">
     Click here to watch the App demo 
  </a>
</p>

### The Osteoarthritis Initiative (OAI) Dataset
<p>The authors thank the <a href="https://nda.nih.gov/oai" target="_blank"> Osteoarthritis Initiative (OAI)</a> for the datasets utilized in this research contribution.</p>


### Publications
+ <p align="justify"> KneeXNet-2.5D: A Clinically-Oriented and Explainable Deep Learning Framework for MRI-Based Knee Cartilage and Meniscus Segmentation <i>(under review at npj Health Systems)</i> </p>



### Acknowledgements
<p align="justify"> We gratefully acknowledge the Department of Health Information Management at the School of Health and Rehabilitation Sciences (SHRS), University of Pittsburgh, for providing the computational infrastructure and support that enabled the training and evaluation of our deep learning models. Their resources and technical assistance were instrumental in the successful completion of this study. </p>

### Citation:

<p align="justify">This contribution is fully described in our manuscript, entitled "<i>KneeXNet-2.5D: A Clinically-Oriented and Explainable Deep Learning Framework for MRI-Based Knee Cartilage and Meniscus Segmentation</i>", which is currently under review at <a href="https://www.nature.com/npjhealthsyst/" target="_blank">npj Health Systems</a>. Any publication or use of this work should cite this manuscript accordingly.</p> 


<p align="center">
  <a href="https://pitthexai.github.io/index.html" target="_blank">
    <img src="Figures/Pitthexai_QR.jpg" alt="Support QR Code" width="200"/>
  </a><br/>
  <b>Pitt Health + Explainable AI (Pitt HexAI) Research Laboratory</b>
</p>


