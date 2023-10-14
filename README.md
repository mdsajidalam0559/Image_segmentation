# image_segmentation

# Image Segmentation using a U-Net

* forest_segmentation: binary segmentation
    * **data: https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation**
    * model: unet_segmentation.py
    * input is expected to be in folders 'masks' and 'images' as .jpg images

* cityscapes-images-paris: multiple segmentation
    * data: https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs
    * preprocessing of the multiclass masks has been adapted from: https://www.kaggle.com/code/yauhenikavaliou/camseq-semantic-segmentation
    * model: unet_segmentation_multi.py
    * input is expected to be in folders 'masks' and images

* Possible improvements:
    * add class weights 
---

# Semantic Image Segmentation using U-Net and CNNs for Self-Driving Cars

## Overview
This repository contains the implementation of a U-Net architecture for semantic image segmentation on the CARLA self-driving car dataset using deep learning and Convolutional Neural Networks (CNNs). Semantic image segmentation is a critical task in autonomous driving, enabling vehicles to understand their surroundings by classifying each pixel in an image into predefined categories.

## Technologies Used
- **Framework:** TensorFlow
- **Architecture:** U-Net
- **Accuracy:** 97%

## Key Features
- **U-Net Architecture:** Implemented a U-Net model from scratch, a widely used architecture in image segmentation tasks due to its effectiveness in capturing fine-grained details.
- **CARLA Dataset:** Utilized the CARLA self-driving car dataset, a diverse and challenging dataset for training and evaluating the segmentation model.
- **Deep Learning:** Leveraged Convolutional Neural Networks (CNNs) to learn spatial hierarchies and features for accurate pixel-wise predictions.
- **High Accuracy:** Achieved an impressive accuracy of 97% in semantic image segmentation, demonstrating the model's ability to accurately classify objects in the environment.

## How to Use
1. **Dataset Preparation:** Download the CARLA self-driving car dataset and preprocess the images and corresponding segmentation masks for training.
2. **Training:** Train the U-Net model using TensorFlow, adjusting hyperparameters as necessary for optimal performance.
3. **Evaluation:** Evaluate the trained model on test data to assess its segmentation accuracy and make any necessary improvements.
4. **Inference:** Use the trained model for real-time or batch inference on new images to perform semantic image segmentation.

## Results
The trained U-Net model exhibited remarkable performance with an accuracy of 97% in classifying objects in the CARLA dataset. The segmentation results demonstrate the model's ability to accurately delineate objects such as vehicles, pedestrians, and road markings, showcasing its potential for real-world applications in self-driving cars.

## Future Work
- **Fine-Tuning:** Explore fine-tuning strategies to further enhance the model's accuracy, especially in challenging scenarios such as adverse weather conditions and complex urban environments.
- **Real-Time Inference:** Optimize the model for real-time inference, ensuring low latency and high throughput to enable seamless integration into autonomous vehicles.
- **Data Augmentation:** Implement advanced data augmentation techniques to augment the training dataset, enhancing the model's ability to generalize to diverse driving scenarios.



---

