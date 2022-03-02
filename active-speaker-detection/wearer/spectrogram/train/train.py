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
import pdb
import scipy.ndimage
import os.path
import scipy.io as sio
import random
import pickle


train_numbers = np.loadtxt('train.txt').astype('int').tolist()

class MyDataset(utils.Dataset):
    def __init__(self):
        self.image_label_file = []

        for n in train_numbers:
           print(n) 
           for m in range(9000):
                if not os.path.isfile('../data/train_test_data/' + str(n) + '/' + str(m) + '.pickle'):
                   continue;

                self.image_label_file.append('../data/train_test_data/' + str(n) + '/' + str(m) + '.pickle')

        print(len(self.image_label_file))


    def __getitem__(self, idx):

        with open(self.image_label_file[idx], 'rb') as handle:
            b = pickle.load(handle)
            im = b['spec']
            im[np.isnan(im)] = 0
            im = im.astype('float32')
            im[im < -100] = -100
            im[im > 100] = 100
            im = cv.resize(im, (255, 255), interpolation=cv.INTER_LINEAR)
            im = np.dstack((im, im, im))
            im = im.swapaxes(0, 2).swapaxes(1, 2)

            label = b['label'] 
            #if label != 0 or label != 1:
            #   label = 0
            label = np.reshape(label, (1,))
            label = label.astype('int64')
            return (torch.Tensor(im), torch.LongTensor(label))

    def __len__(self):
        return len(self.image_label_file)

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
        return x 

net = Res18()
net.cuda()
my_dataset = MyDataset()
my_dataloader = utils.DataLoader(my_dataset, batch_size=50, shuffle=True, num_workers=5) # create your dataloader

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(20):  # loop over the dataset multiple times

    for i, data in enumerate(my_dataloader, 0):
        inputs, labels = data
        labels = labels.squeeze(1)

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = (np.argmax(outputs.data.cpu().numpy(), axis=1) == labels.data.cpu().numpy()).sum() / float(labels.data.cpu().numpy().shape[0])

        running_loss = loss.data.item()

        print('[%d, %5d] loss: %f acc: %f' %(epoch + 1, i + 1, running_loss, acc))

    torch.save(net.state_dict(), 'backup' + str(epoch) + '.model')

print('Finished Training')
torch.save(net.state_dict(), 'final.model')
