#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Summary: Batch convert the people tracking results to the format for MOT benchmarking 
# Usage: python3 track2motformat.py 

import os.path
import cv2 as cv
import numpy as np

MAX_NUMBER_OF_FILES = 10000  # A large number
for fileno in range(MAX_NUMBER_OF_FILES): 
    print(fileno)
    if not os.path.isfile('../../global_tracking/results/' + str(fileno) + '.txt'):
       continue
    res = np.loadtxt('../../global_tracking/results/' + str(fileno) + '.txt', delimiter=' ')
    res = res.astype('int')
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)
    
    mot_res = []
    for n in range(res.shape[0]):
        mot_res.append([res[n][0]+1, res[n][1], res[n][2], res[n][3], res[n][4]-res[n][2], res[n][5]-res[n][3], -1, -1, -1, -1])
    
    np.savetxt('mot_format/' + str(fileno) + '.txt', mot_res, fmt='%d', delimiter=',')
