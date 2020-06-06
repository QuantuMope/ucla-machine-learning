import os
import time
import argparse
from torch import optim
import random as rand
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tvt
import numpy as np
import math
import json
import sys
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import pickle

def initLinear(linear, val = None):
    if val is None:
        fan = linear.in_features +  linear.out_features 
        spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
    else:
        spread = val
        linear.weight.data.uniform_(-spread,spread)
        linear.bias.data.uniform_(-spread,spread)

class resnet_34(nn.Module):
    def __init__(self):
        super(resnet_34, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*512, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 512
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


if __name__ == "__main__":

    model = resnet_34()

    IMG_DATA_DIRECTORY = './preprocessed_testing_img_data.pkl'
    OUTPUT_DIRECTORY = './feature_embedding.csv'
    all_test_img_data = np.load(IMG_DATA_DIRECTORY)
    # with open(IMG_DATA_DIRECTORY, 'rb') as f:
    #     all_test_img_data = pickle.load(f)
    img_data = torch.from_numpy(all_test_img_data)
    count = 0
    for img in img_data:
        #img = torch.from_numpy(img_data)
        print("begin to generate embedding")
        rep = model(img)
        print("saving...")
        with open(OUTPUT_DIRECTORY, 'wb') as f:
            pickle.dump(rep, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(", ", f, protocol=pickle.HIGHEST_PROTOCOL)
        print(count+1)