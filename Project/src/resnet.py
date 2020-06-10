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
        x = x.float()
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


def load_batch(dataset, batch_sz):
    n_batches = (len(dataset) + batch_sz - 1) // batch_sz
    for i in range(0, n_batches):
        yield i, dataset[i * batch_sz : (i + 1) * batch_sz]


IMG_DATA_DIRECTORY = './preprocessed_testing_img_data.pkl'
TEST_DATA_DIRECTORY = "./preprocessed_testing_img_data_extremely_small.pkl"
OUTPUT_DIRECTORY = './feature_embedding.csv'


def extract_feature(dataset_dir, output_dir):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    os.system("rm -rf {}".format(output_dir))
    model = resnet_34().to(device)
    all_test_img_data = np.load(dataset_dir)
    print(all_test_img_data.shape)
    img_data = torch.from_numpy(all_test_img_data)
    for i, img in load_batch(img_data, 32):
        img = img.to(device)
        rep = model(img)
        print("[{}] saving {}".format(i, img.shape))
        # append to previous data
        with open(output_dir, 'ab') as fo:
            data = rep.cpu().detach().numpy()
            pickle.dump(data, fo)


def format_output(output_dir):
    data = []
    with open(output_dir, 'rb') as f:
        while True:
            try:
                x = pickle.load(f)
                data.append(x)
            except:
                break
    data = np.concatenate(data)
    # over-write previous data
    with open(output_dir, 'wb') as fo:
        pickle.dump(data, fo)
    loaded_data = np.load(output_dir)
    assert(data.shape == loaded_data.shape)
    print("Data validation passed :)")


if __name__ == "__main__":
    extract_feature(IMG_DATA_DIRECTORY, OUTPUT_DIRECTORY)
    format_output(OUTPUT_DIRECTORY)