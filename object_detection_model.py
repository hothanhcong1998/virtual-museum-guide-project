import config
import timm
from torchvision.models import resnet50, resnet101
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import cv2
import os
import pandas as pd
from torch.nn import Dropout, Identity, Linear, Module, ReLU, Sequential, Sigmoid


class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
        # initialize the base model and the number of classes
        # if you want to create resnet model, please change classifier to fc
        # if you want to create efficientNet model, please change fc to classifier
        self.baseModel = baseModel
        self.numClasses = numClasses
        self.regressor = Sequential(
                Linear(baseModel.classifier.in_features, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 32),
                ReLU(),
                Linear(32, 4),
                Sigmoid()
            )
        self.classifier = Sequential(
                Linear(baseModel.classifier.in_features, 512),
                ReLU(),
                Dropout(),
                Linear(512, 512),
                ReLU(),
                Dropout(),
                Linear(512, self.numClasses)
            )
        # set the classifier of our base model to produce outputs
        # from the last convolution block
        self.baseModel.classifier = Identity()

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        # return the outputs as a tuple
        return (bboxes, classLogits)


def init_model():
    efficient=timm.create_model('efficientnet_b0', pretrained=False)
    objectDetector = ObjectDetector(efficient, config.NUM_CLASSES)
    #resnet = resnet50(pretrained=False)
    #objectDetector = ObjectDetector(resnet, config.NUM_CLASSES)
    objectDetector.load_state_dict(torch.load(config.CKPT_PATH, map_location=torch.device('cpu')))
    return objectDetector