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

### 1.3 Role of Deep Learning, Specifically [YOLOv8](https://github.com/ultralytics/ultralytics)

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

The methodology employed in this study involved the utilization of deep learning techniques to detect and localize hemorrhage types within head CT scan images.
The CQ500 Head CT images dataset served as the source of annotated CT scan images, which underwent preprocessing to mitigate noise and enhance image quality. Subsequently, a pre-trained deep learning network was employed for training purposes. The training data comprised 80% of the preprocessed images, while the remaining 20% constituted the testing data. The trained deep learning model was then tasked with generating bounding boxes, indicating the location and extent of hemorrhage regions within the CT scan images. The employed method is summarized in following Figure.

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/aafcfdb3-82de-41cc-9e7e-c8d96d86c04c" alt="image07">
</p>

### 2.3 Preprocessing Steps

1. **Pixel Value Transformation to Hounsfield Unit (HU):** Ensures consistent representation.
2. **Windowing:** Adjusts pixel value range for practical display.
3. **Normalization:** Maps pixel values to the range of [0, 255].
4. **Morphology Dilation:** Removes small noise particles.
5. **Dimension Squeezing:** Reduces unnecessary dimensions.

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/21f294ca-808d-4532-8d73-f157fcf571bc" alt="image08">
</p>
<p align="center">
  (a) Before pre-process    (b) After pre-process
</p>

### 2.4 Data Preparation for Training

In the preparation phase of the study, it was noted that certain images in the CQ500 dataset contained multiple instances of brain hemorrhages, each corresponding to a different type. To address this, a data preprocessing strategy was implemented, involving the separation of individual annotations for each type of hemorrhage within such images. This approach facilitated the introduction of the same image multiple times into the training pipeline, with each iteration dedicated to a distinct annotation. This strategy ensured that the deep learning model could effectively discern and annotate each type of hemorrhage independently, even within the same image context. Despite the need for repetitive presentation of identical images with distinct annotations, this method significantly contributed to enhancing the model's capacity to accurately detect and classify various types of brain hemorrhages, particularly when they co-occur within a single image. The analysis of class imbalance, as depicted in following figure, revealed that it was not a significant issue, except for the Epidural class. The final dataset for training and validation consisted of 21,751 images (80%) and 5,438 images (20%), respectively. Additionally, since YOLOv8 was utilized as the model, image resizing from **_512x512 to 640x640_** pixels was performed to align with the model's requirements. This comprehensive preprocessing strategy and dataset preparation set the foundation for the subsequent successful training and evaluation of the YOLOv8 model in the detection and localization of intracranial hemorrhages.


<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/ef0ed972-56c1-4287-9254-376c9ec72604" alt="image09">
</p>

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/edf0b2e2-9e95-447a-bb5d-21326550e0dc" alt="image10">
</p>


### 2.5 [YOLOv8](https://github.com/ultralytics/ultralytics)

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/8800ddc8-de27-4291-b15f-220774e1338c" alt="image11">
</p>

[YOLOv8](https://github.com/ultralytics/ultralytics) is a state-of-the-art object detection algorithm that offers exceptional performance in terms of accuracy and speed. It's the latest iteration in the YOLO series, built by Ultralytics. YOLOv8 stands out for its advancements in deep learning and computer vision, providing unparalleled capabilities for object detection, segmentation, pose estimation, tracking, and classification.

**Key Features of [YOLOv8](https://github.com/ultralytics/ultralytics):**
- **Real-time Object Detection:** YOLOv8 excels in processing images and videos in real-time, making it suitable for applications that require immediate object identification.

- **High Accuracy:** YOLOv8 achieves impressive accuracy in detecting and classifying objects, even in complex environments with varying lighting conditions and occlusions.

- **Scalability:** YOLOv8 can be scaled to different performance requirements by adjusting its backbone network and input image resolution.

- **Versatility:** YOLOv8 supports a wide range of object detection tasks, including instance segmentation, person detection, and vehicle detection.

- **Ease of Use:** YOLOv8 provides a user-friendly interface and extensive documentation, making it accessible to both beginners and experienced developers.

**Applications of [YOLOv8](https://github.com/ultralytics/ultralytics):**

- Autonomous Vehicles: YOLOv8 can detect and track objects on the road, enabling autonomous vehicles to navigate safely.

- Surveillance Systems: YOLOv8 can be used in surveillance systems to monitor for suspicious activity and identify potential threats.

- **Medical Image Analysis:** YOLOv8 can assist in medical image analysis by detecting and classifying abnormalities in **_X-rays and CT scans_**.

- Retail Analytics: YOLOv8 can analyze customer behavior in retail stores to understand product interactions and optimize store layouts.

- Agricultural Monitoring: YOLOv8 can monitor crop health and identify pests or diseases in agricultural fields.

[YOLOv8](https://github.com/ultralytics/ultralytics) is employed for hemorrhage detection, utilizing its advanced architecture, self-attention mechanism, multi-scaled object detection, and ensemble approach.

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/7800cf12-8002-431a-bd8b-37d8b03df4f9" alt="image12">
</p>

## Results and Discussions

### 3.1 Employed Parameters

- The hemorrhage detection model was developed using the CQ500 dataset, consisting of annotated CT scan images.
- Training was conducted incrementally with a growing number of epochs due to the dataset's size.
- Pretrained weights initialized the detection network, followed by fine-tuning for hemorrhage detection.
- **_Stochastic gradient descent (SGD)_** with the **_Adam optimizer_** was employed, starting with a **_learning rate of 0.01_** and **_batch sizes of 16_**.
- Training concluded when the loss function converged.
- The model architecture comprises **_225 layers_** with **_3,012,018 learnable parameters and gradients_**.
- The training and evaluation were performed on **_Kaggle GPU T4 x2 instances with 16GB RAM_** in a cloud-based computing environment.

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/5735a054-5368-4eba-954c-4530ed600521" alt="image13">
</p>

### 3.2 Detecting Bounding Boxes and Hemorrhage Types

YOLOv8 successfully detects multiple hemorrhages in a single scan, showcasing its potential for clinical applications.
Despite using a single label per image during training, the model demonstrated the capability to accurately predict multiple labels within a single scan, as illustrated in following figure. The results, as depicted in following figure, showcased the successful prediction of multiple bounding boxes for hemorrhages. This outcome is noteworthy as it highlights YOLOv8's effectiveness in detecting multiple hemorrhages within a single scan, even when the training dataset initially had images with only a single label. This capability holds significance for clinical applications where the identification of all present hemorrhages in an image is crucial. For further insights, the confusion matrix is recommended.

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/1244da8b-4e7e-402e-a52e-4e68a34d876c" alt="image14">
</p>

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/8e3ea2ea-7051-47d9-ac23-2cd1ee887c10" alt="image15">
</p>

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/6b7b4036-2840-4805-8333-ddface3fc5b0" alt="image16">
</p>

### 3.3 Comparison

The precision of the model demonstrated improvement with an increase in the number of trained epochs. However, due to limitations in available GPU resources, the model could not be trained for the desired number of epochs. The authors suggest that achieving even higher precision is plausible by extending the training duration. The provided plots offer a summary of the obtained results for the model evaluation.

**_Note: all plots contains details about 250 - 350 epochs._**
<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/d0922e31-79df-4f58-b9b9-7379184ee8b9" alt="image17">
</p>

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/6bc4a495-1aed-4d4c-a89e-9f55da0a9b81" alt="image18">
</p>

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/b9ee95f5-6099-4f91-a22a-55c3b5c97a46" alt="image19">
</p>

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/3e2703ae-6476-4262-bcbe-d0b0e80cf0bd" alt="image20">
</p>

<p align="center">
  <img src="https://github.com/RavinduMPK/Deep-Learning-Based-Detection-and-Localization-of-Intracranial-Hemorrhage-Types/assets/68577937/a97c23d6-5664-4a17-beb9-35d3d46567b6" alt="image21">
</p>



## Future Work

Looking ahead, we're planning to make our model even better. We'll start by fine-tuning how we process images, tweaking the model's design, and playing around with settings to get the best performance possible. 
But that's not all – we're also exploring new methods that could take our accuracy to the next level.

Imagine this: we're extending our training sessions, giving the model more time to learn and improve. And here's the kicker – we're not just stopping at accuracy.
We're working hard to create a dataset that's fair and covers all the bases when it comes to different types of hemorrhages. The endgame? We're aiming for error rates so low they'll make you do a happy dance. So, get ready for what's next – because the future of our model looks pretty exciting!

## Conclusion

The project demonstrates the potential of YOLOv8 for detecting and localizing intracranial hemorrhage types, showcasing improved diagnostic accuracy for timely patient management.

## Acknowledgment

We express our appreciation to **Dr. Ranga Rodrigo** and Mr. **Tharindu Wickremasinghe** for their invaluable guidance and insights.

## References

1. Ömer Faruk Ertuğrul, Muhammed Fatih Akıl, "Detecting hemorrhage types and bounding box of hemorrhage by deep learning," Biomedical Signal Processing and Control, Volume 71, Part A, 2022, 103085.
2. PhysioNet. (2020). BHX: Brain Hemorrhage Extended (BHX): Bounding box extrapolation from thick to thin slice CT images. [Link](https://physionet.org/content/bhx-brainbounding-box/1.1/)
3. Heit, J. J., Iv, M., & Wintermark, M. (2017). "Imaging of Intracranial Hemorrhage." J Stroke, 19(1), 11–27. doi: 10.5853/jos.2016.00563.
4. OpenMMLab. (2023, January 17). "Dive into YOLOv8: How does this state-of-the-art model work?" Medium. [Link](https://openmmlab.medium.com/dive-into-yolov8-how-does-this-state-of-the-art-modelwork-10f18f74bab1#:~:text=In%20summary%2C%20YOLOv8%20is%20a,object%20detection%2C%20and%20instance%20segmentation)
