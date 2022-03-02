#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import os.path
import json

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import cv2 as cv
import pickle
import sys

class HeadFeatureNet(nn.Module):
    def __init__(self, output_num = 128):
        super(HeadFeatureNet, self).__init__()
        model_reg = models.resnet18(pretrained=True)
        self.model = model_reg
        self.f = nn.Linear(in_features=512, out_features=output_num, bias=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.f(x)
        return x

net = HeadFeatureNet()
net.load_state_dict(torch.load('../../models/head_feature.model'))
net.eval()
net.cuda()

with open('v.txt') as f:
    content = f.readlines()
filenames = [x.strip() for x in content]

datadir = sys.argv[1]
fileno = int(sys.argv[2]) 
res = np.loadtxt('../short_term_tracking/build/result.txt', delimiter=',')
if len(res.shape) == 1:
    res = np.expand_dims(res, axis=0)

box = {};

for n in range(res.shape[0]):
    if not (res[n][0] in box):
       box[res[n][0]] = []
    box[res[n][0]].append([res[n][1], res[n][2], res[n][3], res[n][4]+res[n][2], res[n][5]+res[n][3]]);

fname = datadir + '/' + filenames[fileno]
feature = {}

cap = cv.VideoCapture(fname)
frame_num = 0

sys.stdout.write('\033[K')
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    print(frame_num, end = '\r')
    if frame_num in box:
       b = box[frame_num]
       for k in range(len(b)):
           pid = int(b[k][0])
           x1 = int(b[k][1])
           y1 = int(b[k][2])
           x2 = int(b[k][3])
           y2 = int(b[k][4])

           if x1 < 0:
              x1 = 0
           if y1 < 0:
              y1 = 0
           if x2 >= frame.shape[1]:
              x2 = frame.shape[1]-1
           if y2 >= frame.shape[0]:
              y2 = frame.shape[0]-1 

           if x1 >= x2 or y1 >= y2:
              continue

           if pid not in feature:
               feature[pid] = []
           imcrop = frame[y1:y2,x1:x2,:]
           imcrop = cv.resize(imcrop, (128,128))        
           imcrop = imcrop.swapaxes(0, 2).swapaxes(1, 2)
           imcrop = (imcrop.astype('float32')/255.0)
           imcrop = imcrop.reshape((1,3,128,128))
           imcrop = Variable(torch.Tensor(imcrop))
           output = net(imcrop.cuda())
           output = output.data.cpu().numpy().squeeze().tolist()

           feature[pid].append([frame_num, x1, y1, x2, y2] + output)

    frame_num = frame_num + 1       


with open('traj.pkl', 'wb') as f:
      pickle.dump(feature, f, pickle.HIGHEST_PROTOCOL)
