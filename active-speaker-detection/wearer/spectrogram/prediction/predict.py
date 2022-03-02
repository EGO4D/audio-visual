#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 predict.py [test|val]

import numpy as np
import torch
import torch.utils.data as utils
import torchvision.models as models
import cv2 as cv
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb
import scipy.ndimage
import os.path
import scipy.io as sio
import random
import pickle
from sklearn.metrics import average_precision_score
import sys


class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()
        self.model1 = models.resnet18(pretrained=True)
        self.f1 = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):
        x = self.model1.conv1(x)
        x = self.model1.bn1(x)
        x = self.model1.relu(x)
        x = self.model1.maxpool(x)
        x = self.model1.layer1(x)
        x = self.model1.layer2(x)
        x = self.model1.layer3(x)
        x = self.model1.layer4(x)
        x = self.model1.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.f1(x)
        x = nn.Softmax(dim=1)(x)
        return x 

net = Res18()
net.load_state_dict(torch.load('../../../../models/wearer_spectrogram_classification.model'))
net.eval()
net.cuda()

data_set = sys.argv[1] # 'test' or 'val'

if not os.path.isfile(data_set + '.txt'):
   sys.exit(data_set + ' list does not exist.')

test_nums = np.loadtxt(data_set + '.txt')
test_nums = test_nums.astype('int').tolist()

for n in test_nums: 
   y = np.zeros(9000) 
   for m in range(9000):
        print(n, m)
        if not os.path.isfile('../data/train_test_data/' + str(n) + '/' + str(m) + '.pickle'):
           continue;

        image_label_file = '../data/train_test_data/' + str(n) + '/' + str(m) + '.pickle'

        with open(image_label_file, 'rb') as handle:
            b = pickle.load(handle)
            im = b['spec']
            im[np.isnan(im)] = 0
            im = im.astype('float32')
            im[im < -100] = -100
            im[im > 100] = 100
            im = cv.resize(im, (255, 255), interpolation=cv.INTER_LINEAR)
            im = np.dstack((im, im, im))
            im = im.swapaxes(0, 2).swapaxes(1, 2)
            im = np.expand_dims(im, 0)
        
            label = b['label']
        
            inputs = Variable(torch.Tensor(im).cuda())
            outputs = net(inputs)
            p = outputs.cpu().detach().numpy()
            p = p[0]

            y[m] = p[1]

   np.savetxt('../../results/' + str(n) + '.txt', y, fmt='%f')
            

