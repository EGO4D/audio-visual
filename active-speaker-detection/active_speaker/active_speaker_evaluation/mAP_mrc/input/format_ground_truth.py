#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 format_ground_truth.py data_set

import os.path
import cv2 as cv
import numpy as np
import sys

with open('v.txt') as f:
    content = f.readlines()
filenames = [x.strip() for x in content]

data_set = sys.argv[1] # 'test' or 'val'

if data_set == 'test':
   sys.exit('No test set annotations.')

vlist = np.loadtxt(data_set + '.txt')
vlist = vlist.astype('int').tolist()

sys.stdout.write('\033[K')

for fileno in vlist:
    print(fileno, end = '\r')
    fn = (filenames[fileno]).split('.')[0]

    res = np.loadtxt('../../../../../utils/ground_truth/headbox_wearer_speaker/' + fn + '.txt', delimiter=' ')
    res = res.astype('int')
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)
    
    box = {};
    for n in range(res.shape[0]):
        if not (res[n][0] in box):
           box[res[n][0]] = []
        box[res[n][0]].append([res[n][1], res[n][2], res[n][3], res[n][4], res[n][5], res[n][6]]);
        
    for frame_num in range(9000):
        asp = []
        if frame_num in box:
           b = box[frame_num]
           for k in range(len(b)):
               pid = int(b[k][0])
               x1 = int(b[k][1])
               y1 = int(b[k][2])
               x2 = int(b[k][3])
               y2 = int(b[k][4])
               speak = int(b[k][5]);
               if speak == 1:
                  asp.append([1, x1, y1, x2, y2])
                  
        np.savetxt('ground-truth/' + str(fileno) + '_' + str(frame_num) + '.txt', asp, fmt='%d', delimiter=' ')
