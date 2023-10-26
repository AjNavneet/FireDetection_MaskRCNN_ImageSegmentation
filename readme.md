# Fire detection using Image Segmentation with Mask R-CNN

## Aim

This project aims to make predictions over images and detect mask and bounding boxes around the area of interest, i.e., mask and bounding around the fire in a given
image.

---

## Tech stack
- Language - `Python`
- Libraries – `numpy`, `keras`, `tensorflow`, `mrcnn`, `scikit-image`, `etc (as per mentioned in the requirements.txt file)`

---

## Image Localization and Segmentation

Image localization and segmentation are two important tasks in computer vision. 

- Image localization refers to the process of finding the location of an object of interest in an image, while image segmentation involves dividing an image into multiple segments, where each segment corresponds to a different object or background.

---

## Mask R-CNN

Mask R-CNN (Regional Convolutional Neural Network) is a deep learning architecture that is specifically designed for object detection and instance segmentation. 

It extends the Faster R-CNN architecture by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. 

The network uses a convolutional neural network (CNN) to extract features from the image and uses a region proposal network (RPN) to propose regions that are likely to contain an object.

The mask branch of the network then outputs a binary mask for each instance of the object, allowing for precise instance-level segmentation.

In this project, the Mask R-CNN architecture will be trained on a dataset of fire images, and the network will be evaluated on new images to assess its performance in detecting and localizing fire instances. The final output of the network will be a set of bounding boxes around fire instances, along with a binary mask for each instance, providing a precise representation of the location and extent of fire in the image.

---

### Note

For this project, we are using a wrapper for Mask R-CNN from the matterport repository [here](https://github.com/matterport/Mask_RCNN). However, the original implementation is not compatible with the latest libraries. Therefore, we are using a modified version of the Mask R-CNN project that has been developed by leekunhee [here](https://github.com/leekunhee/Mask_RCNN) and is compatible with TensorFlow 2.0 for our project.

We made a few changes to get it running on Google Colab. The original leekunhee code in mrcnn/model.py at line 2364 was changed to:

```python
# workers=0
# multiprocessing=False
```

---

## Getting Started

### Step 1: Custom Configuration

In this step, we will create a custom configuration file to specify the parameters for training and inference of the Mask R-CNN model. The parameters include information such as the number of classes and batch size, among others.

### Step 2: Custom Data

In this step, we will create a custom dataset of images with annotations for fire instances. The annotations include the bounding boxes and masks for each instance in the images.

### Step 3: Training

In this step, we will use the custom configuration and data to train the Mask R-CNN model. The model will learn to identify and locate fire instances in images and generate masks for them.

### Step 4: Inference

In this step, we will use the trained model to perform inference on new images. The model will be able to localize and segment fire instances in the images, allowing for early fire detection.

### Note

The training is done for just 10 steps per epoch for 20 epochs for demo purposes, and the model is stored in `output/logs/object20230207T1931/mask_rcnn_object_0020.h5`. Please train for more accurate results.

## Execution Instructions

1. Upload the code folder to Google Drive and open the Jupyter file with Google Colab.
2. Mount your Google Drive.
3. Change the working directory to the root folder of the project.
4. Install the required packages from the `requirements.txt` file using `pip install -r requirements.txt`.
5. Change the working directory to the `Mask_RCNN` folder.
6. Update the configurations, such as datasets, the number of steps per epoch, and other relevant parameters in the code.
7. Train the model using the updated parameters.
8. Alternatively, if you'd like to load a previously trained model, you can load it from the last section of the Jupyter Notebook for inference.

Note: The requirements file should be from the root folder, not the leekunhee Mask_RCNN repository.

---