# CocoaGuard Machine Learning Pipeline

This folder contains the complete machine learning pipeline for the CocoaGuard AI engine. The workflow is split across four sequential Jupyter Notebooks:

## 1. 01_train_efficientnetv2b0.ipynb
This is the core engine room. It handles loading the dataset, augmenting the images, and running the two-phase training process for the EfficientNetV2-B0 Convolutional Neural Network.

## 2. 02_evaluate_and_deploy.ipynb
This is the grading phase. It takes the trained model and tests it against unseen data to generate accuracy scores, classification reports, and the crucial confusion matrix. It then saves the finalized model.

## 3. 03_single_image_test.ipynb
Think of this as a quick sandbox. It is a utility script used to pass just one single image through the trained model to manually verify that the predictions are working correctly before integrating it into an app.

## 4. 04_tflite_model_conversion.ipynb
This is the compression stage. It takes the heavy, finalized model and uses post-training quantization to shrink it down into a lightweight TensorFlow Lite (.tflite) file so it can run fast and offline.
