#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 batch_audio_embedding.py directory_of_ego4d_videos data_set

import numpy as np
import torch
import torch.utils.data as utils
import torchvision.models as models
import cv2 as cv
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os.path
import random
import torchvision as tv
import audiosegment
import time
import sys
import pickle


def specgram(y, Fs):
    window = int(25*Fs/1000)
    step = int(5*Fs/1000)
    mask = np.hamming(window)
    spec = []
    for n in range(y.shape[0], 0, -step):
        if n-window >= 0:
           z = y[n-window:n] * mask
           s = np.absolute(np.fft.fft(z, 512))
           spec.append(s)

    spec = np.array(spec)
    spec = np.log10(spec + 1e-10)
    return spec

class Resa(nn.Module):
    def __init__(self, output_num = 128):
        super(Resa, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.f = nn.Linear(in_features=512, out_features=output_num, bias=True)
        self.conv1 = self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
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

video_dir = sys.argv[1]
data_set = sys.argv[2]

if not os.path.isfile(data_set + '.txt'):
   sys.exit(data_set + ' list does not exist.')

with open('v.txt') as file:
    lines = file.readlines()
    video_files = [line.rstrip() for line in lines]

video_nums = np.loadtxt(data_set + '.txt')  # test.txt (test video ids), val.txt (validation video ids)
video_nums = video_nums.astype('int').tolist()

neta = Resa()
neta.load_state_dict(torch.load('../../../models/audio_embedding.model'))
neta.cuda()

for video_id in video_nums:
     vname = video_dir + '/' + video_files[video_id]
     cap = cv.VideoCapture(vname)
     video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
     video_fps =  cap.get(cv.CAP_PROP_FPS)
     cap.release()
     
     sound = audiosegment.from_file(vname)
     sound = sound.resample(sample_rate_Hz=16000, sample_width=2, channels=1)
     samples = sound.get_array_of_samples()
     fps = 16000
     aud = np.array(samples).astype('float32')/65536.0
     total_time = int(aud.shape[0]/fps)
     #print('audio time', total_time, aud.shape[0], video_frames, video_frames/video_fps)
     
     ff = []
     sys.stdout.write('\033[K')
     for frame_num in range(video_frames):
         print(frame_num, end = '\r')
         afirst = int(frame_num * 1.0 / float(video_fps) * fps - 0.5*fps)
         alast = int(afirst + fps)
     
         if afirst < 0:
            afirst = 0
        
         if alast >= aud.shape[0]:
            alast = aud.shape[0]-1 
         
         y = aud[afirst:alast]
     
         sp = specgram(y, 16000)
         sp = sp[:,0:256]
         sp = np.reshape(sp, (1,1,sp.shape[0],sp.shape[1]))
         
         ainput = torch.Tensor(sp)
         ainput = Variable(ainput.cuda())
         
         f = neta(ainput)
         f = f.cpu().detach().numpy()
     
         ff.append(f[0]) 
         
     with open('embeddings/' + str(video_id) + '.pickle', 'wb') as handle:
         pickle.dump(ff, handle, protocol=pickle.HIGHEST_PROTOCOL)    




