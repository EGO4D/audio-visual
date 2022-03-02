#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 format_asd.py asd_results_dir data_set

import os
import cv2 as cv
import numpy as np
import sys

asd_results_dir = sys.argv[1]
data_set = sys.argv[2]

if not os.path.isfile(data_set + '.txt'):
   sys.exit(data_set + ' list does not exist.')

with open('v.txt') as f:
    content = f.readlines()
filenames = [x.strip() for x in content]


vlist = np.loadtxt(data_set + '.txt')
vlist = vlist.astype('int').tolist()

sys.stdout.write('\033[K')

for fileno in vlist:
    print(fileno, end = '\r')
    res = np.loadtxt(asd_results_dir + '/' + str(fileno) + '.txt', delimiter=' ')
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)
    
    box = {};
    for n in range(res.shape[0]):
        if not (res[n][0] in box):
           box[res[n][0]] = []
        box[res[n][0]].append([res[n][1], res[n][2], res[n][3], res[n][4], res[n][5], res[n][6], res[n][7]]);
        

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
               speakf = float(b[k][5])
               speak = int(b[k][6])
               asp.append([1, speakf, x1, y1, x2, y2])

        if len(asp) > 0:       
            np.savetxt('detection-results/' + str(fileno) + '_' + str(frame_num) + '.txt', asp, fmt='%d %f %d %d %d %d', delimiter=' ')
        else:
            np.savetxt('detection-results/' + str(fileno) + '_' + str(frame_num) + '.txt', asp, fmt='%d', delimiter=' ')
            
