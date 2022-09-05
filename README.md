# cifia-10-test
This is my test about classifying cifia-10.In this project, I created dataset and dataloader and did some random transformations of the original dataset.Then I created VGG, ResNet, MobileNet, and InceptionNet to train on.

data_pre.py In this document, we divide the cifai-10 data into train and test categories. Then in each class, it is divided according to 10 sub-classes.
model.py In this file, we implement models such as VGG, ResNet, MobileNet, and InceptionNet.
train.py In this file, you can use one of the models for training and prediction.