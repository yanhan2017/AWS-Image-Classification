# Autonomous Inventory Monitoring at Distribution Centers

For distribution centers, it is crucial to have an accurate record of its inventory. However, counting the inventory manully is a very time consuming process. This project tries to automate this process by building a deep learning model that takes in an image of a bin and outputs the number of items in the bin.

## Project Set Up and Installation

To run this project, you need access to AWS sagemaker. Create a notebook instance with ml.t3.medium instance and upload sagemaker.ipynb, train.py and file_list.json files. Open sagemaker.ipynb with conda_pytorch_p39 kernel. In addition to the pre-installed libraries, you also need to install split-folders and smdebug. The code for installation is already included in the jupyter notebook.

## Dataset

### Overview

The dataset used is derived from the Amazon Bin Image Dataset. It contains over 10,000 RGB images from an operating Amazon Fulfillment Center and each image has label 1 - 5. 

### Access

The images are first downloaded into the workspace, organized into different folders based on their label and divided into training, validation and test sets. For example, if an image has label of 3 and belongs to the test set, it will be stored in folder /test/1. The data folders are then uploaded to S3 for training.

## Model Training

Several pretrained models and hyperparameter settings are attempted to find the best performing model. Most weights from the pretrained models are reused, and only their last fully connected layer is retrained on the bin image dataset. The model architectures attempted are ResNet18 and ResNet34. The hyperparameters tuned are learning rate and training batch size. There are six total trainings conducted and the training with the smallest test loss has ResNet18 architecture, learning rate 0.0001225 and batch size 128. With 15 epochs of training it achieves final test accuracy of 32.412%.


## Machine Learning Pipeline

This project first does data exploration manually. Data preprocessing is done automatically when loading data into the model. Hyperparameter tuning and model selection is also done automatically.