# Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/ac7e4bf9-c715-46eb-ae2a-7d2cc3851c4b" alt="image01">
</p>

## Introduction

### 1.1 Intracranial Hemorrhage (ICH): A Critical Medical Event

Intracranial Hemorrhage (ICH) is a critical medical condition characterized by bleeding within the skull. It holds substantial clinical significance, accounting for up to 15% of all strokes. Strokes, a severe vascular event affecting the brain, have various causes, and ICH represents a significant subset with distinct clinical implications.

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/43f657cf-9491-46f3-8ad1-ec3aa58b2076" alt="image02">
</p>


### 1.2 Importance of Timely Detection and Classification

Timely detection and accurate classification of ICH are paramount for effective medical intervention. Given the potential severity of intracranial hemorrhages and their contribution to strokes, early identification enables prompt medical responses, leading to improved patient outcomes. Non-contrast head CT scans serve as a crucial diagnostic tool for swiftly assessing and diagnosing ICH.

### 1.3 Role of Deep Learning, Specifically YOLOv8

This project emphasizes the application of deep learning, with a specific focus on YOLOv8, to address the challenges associated with detecting and classifying various types of intracranial hemorrhages. Deep learning algorithms, and in this case, YOLOv8, have demonstrated remarkable efficacy in image analysis tasks, especially in medical imaging. YOLOv8's ability to accurately identify and localize hemorrhages in CT scans showcases its potential for enhancing diagnostic processes and facilitating timely clinical interventions.

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/5caac976-11d5-4531-98ee-0970eaf80989" alt="image03">
</p>


## Materials and Methods

### 2.1 Dataset

The BHX dataset is a publicly available dataset that serves as an extension of the qure.ai CQ500 dataset, specifically focusing on intracranial hemorrhages. This dataset is designed to facilitate research and development in the field of medical image analysis, particularly for the detection and localization of different types of acute hemorrhages within the brain.


**Source and Origin:**
- The BHX dataset is an extension of the **_qure.ai CQ500_** dataset. The CQ500 dataset is a collection of anonymized head CT scans that was made available for research purposes.

**Hemorrhage Types:**
- The dataset is annotated for **five** types of acute hemorrhages: **_Intraparenchymal, Subarachnoid, Intraventricular, Epidural, and Subdural_**.
- Additionally, there is an extra **sixth** label for _**Chronic Subdural Hematoma**_.

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/179dc2b3-3d67-4aa1-bb42-d54d95b36ec9" alt="image04">
</p>

**Bounding Box Annotations:**
- The dataset provides bounding box annotations for each of the hemorrhage types. A bounding box typically consists of coordinates specifying the rectangular region in an image where a particular hemorrhage is located.
- There are a total of **_39,668 bounding boxes_** annotated in **_23,409 images_**, indicating the presence and location of hemorrhages within these images.
  
<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/d2817256-375c-4d72-a9d5-96239a51262c" alt="image05">
</p>

**Purpose:**
- The primary purpose of the BHX dataset is to support the development and evaluation of algorithms for the automatic detection and classification of intracranial hemorrhages.
- It enables researchers, data scientists, and developers to train and test machine learning models, particularly deep learning models, for medical image analysis.

**Accessibility:**
- As a public dataset, BHX is accessible to the research community, allowing for collaboration, benchmarking, and advancements in the field of medical imaging. The dataset can be accessed on [PhysioNet](https://physionet.org/content/bhx-brain-bounding-box/1.1/).

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/ef944712-683f-4c38-845c-e457806d3d0a" alt="image06">
</p>


### 2.2 Methodology

The study employs a two-phase methodology involving preprocessing and YOLOv8 model training. The CQ500 Head CT images dataset serves as the source.

### 2.3 Preprocessing Steps

1. **Pixel Value Transformation to Hounsfield Unit (HU):** Ensures consistent representation.
2. **Windowing:** Adjusts pixel value range for practical display.
3. **Normalization:** Maps pixel values to the range of [0, 255].
4. **Morphology Dilation:** Removes small noise particles.
5. **Dimension Squeezing:** Reduces unnecessary dimensions.

### 2.4 Data Preparation for Training

Data preparation addresses class imbalances, ensuring the model handles multifaceted scenarios in medical image analysis.

### 2.5 YOLOv8

YOLOv8 is employed for hemorrhage detection, utilizing its advanced architecture, self-attention mechanism, multi-scaled object detection, and ensemble approach.

## Results and Discussions

### 3.1 Employed Parameters

The model is trained using the CQ500 dataset, incrementally increasing epochs with stochastic gradient descent.

### 3.2 Detecting Bounding Boxes and Hemorrhage Types

YOLOv8 successfully detects multiple hemorrhages in a single scan, showcasing its potential for clinical applications.

### 3.3 Comparison

Precision increases with the number of epochs, limited by available GPU resources.

## Future Work

Future work includes further preprocessing, fine-tuning model architecture, optimizing parameters, and addressing class imbalances for improved accuracy.

## Conclusion

The project demonstrates the potential of YOLOv8 for detecting and localizing intracranial hemorrhage types, showcasing improved diagnostic accuracy for timely patient management.

## Acknowledgment

We express our appreciation to Dr. Ranga Rodrigo and Mr. Tharindu Wickremasinghe for their invaluable guidance and insights.

## References

1. Ömer Faruk Ertuğrul, Muhammed Fatih Akıl, "Detecting hemorrhage types and bounding box of hemorrhage by deep learning," Biomedical Signal Processing and Control, Volume 71, Part A, 2022, 103085.
2. PhysioNet. (2020). BHX: Brain Hemorrhage Extended (BHX): Bounding box extrapolation from thick to thin slice CT images. [Link](https://physionet.org/content/bhx-brainbounding-box/1.1/)
3. Heit, J. J., Iv, M., & Wintermark, M. (2017). "Imaging of Intracranial Hemorrhage." J Stroke, 19(1), 11–27. doi: 10.5853/jos.2016.00563.
4. OpenMMLab. (2023, January 17). "Dive into YOLOv8: How does this state-of-the-art model work?" Medium. [Link](https://openmmlab.medium.com/dive-into-yolov8-how-does-this-state-of-the-art-modelwork-10f18f74bab1#:~:text=In%20summary%2C%20YOLOv8%20is%20a,object%20detection%2C%20and%20instance%20segmentation)
