i#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 train.py 

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

class MyDataset(utils.Dataset):
    def __init__(self):
        self.person = []
        self.audio_num = []

        with open('persons_ids.txt') as f:  # person_ids.txt include all the person ids in voxceleb2. 
                                            # They are also the subdirections in the audios
            content = f.readlines()
        p_names = [x.strip() for x in content]
     
        for vn in range(len(p_names)):
           vdir = 'audios/' + p_names[vn]
           num_files = len([name for name in os.listdir(vdir) if os.path.isfile(os.path.join(vdir, name))])
           if num_files > 5:
              self.audio_num.append(num_files)
              self.person.append(p_names[vn])

    def get_audio(self, fname):
        try: 
            sound = audiosegment.from_file(fname)
            sound = sound.resample(sample_rate_Hz=16000, sample_width=2, channels=1)
            samples = sound.get_array_of_samples()
            fps = 16000
            aud = np.array(samples).astype('float32')/65536.0
            if fps+5 <= aud.shape[0]-5:
                audio_sample_num = random.randint(fps+5, aud.shape[0]-5)
                afirst = audio_sample_num - fps + 1
                alast = audio_sample_num            
            else:
                afirst = 0
                alast = aud.shape[0]-1

            y = aud[afirst:alast]
            sp = specgram(y, 16000)
            sp = sp[:,0:256]
            sp = np.expand_dims(sp, axis=0)
            bad_frame = False
            if sp.shape != (1,195,256):
               sp = np.zeros((1,195,256))
               bad_frame = True
            return (sp, bad_frame)    
        except:
            sp = np.zeros((1,195,256))
            bad_frame = True
            return (sp, bad_frame)


    def __getitem__(self, idx):
        k1, k2 = random.sample(range(len(self.person)), 2)
        a = random.randint(1, self.audio_num[k1])
        b = random.randint(1, self.audio_num[k1]) 
        c = random.randint(1, self.audio_num[k2])

        while(not os.path.isfile('audios/' + self.person[k1] + '/' + str(a) + '.mp4') or
              not os.path.isfile('audios/' + self.person[k1] + '/' + str(b) + '.mp4') or
              not os.path.isfile('audios/' + self.person[k2] + '/' + str(c) + '.mp4')):
                  k1, k2 = random.sample(range(len(self.person)), 2)
                  a = random.randint(1, self.audio_num[k1])
                  b = random.randint(1, self.audio_num[k1])
                  c = random.randint(1, self.audio_num[k2])

        sp1, badf1 = self.get_audio('audios/' + self.person[k1] + '/' + str(a) + '.mp4') 
        sp2, badf2 = self.get_audio('audios/' + self.person[k1] + '/' + str(b) + '.mp4')
        if badf1 or badf2:
           sp1 = np.zeros((1,195,256)) 
           sp2 = np.zeros((1,195,256))

        sp3, badf3 = self.get_audio('audios/' + self.person[k2] + '/' + str(c) + '.mp4')
        if badf3:
            sp3 = np.ones((1,195,256))

        return (torch.Tensor(sp1), torch.Tensor(sp2), torch.Tensor(sp3))

    def __len__(self):
        return 10000 


def triple_loss(f1, f2, f3):
    distance_positive = (f1 - f2).pow(2).sum(1) 
    distance_negative = (f1 - f3).pow(2).sum(1)
    margin = 1
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()


neta = Resa()
neta.cuda()

my_dataset = MyDataset()
my_dataloader = utils.DataLoader(my_dataset, batch_size=5, shuffle=True, num_workers=50)

optimizer = optim.Adam(neta.parameters(), lr=0.00001)

for epoch in range(500): # 500 is the max iterations
    for i, data in enumerate(my_dataloader, 0):
        input1, input2, input3 = data
        input1, input2, input3 = Variable(input1.cuda()), Variable(input2.cuda()), Variable(input3.cuda())

        optimizer.zero_grad()

        f1 = neta(input1)
        f2 = neta(input2)
        f3 = neta(input3)

        loss = triple_loss(f1, f2, f3)
        loss.backward()

        optimizer.step()

        running_loss = loss.data.item()
        print('[%d, %5d] loss: %f' %(epoch + 1, i + 1, running_loss))

    torch.save(neta.state_dict(), 'backup.model')
    
print('Finished Training')
torch.save(neta.state_dict(), 'final.model')

