#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 make_train_test_data.py 

import os.path
import cv2 as cv
import numpy as np
import scipy.io as sio
import pickle

with open('v.txt') as f:
    content = f.readlines()
filenames = [x.strip() for x in content]

for fileno in range(len(filenames)):
    print(fileno, end = '\r')

    if not os.path.isfile('spectrograms/' + str(fileno) + '.mat'):
       continue

    spec = sio.loadmat('spectrograms/' + str(fileno) + '.mat')
    spec = spec['S']
    
    fn = os.path.basename(filenames[fileno]).split('.')[0]
    if not os.path.isfile('../../../../utils/ground_truth/headbox_wearer_speaker/' + fn + '_w.txt'):
        continue
    res = np.loadtxt('../../../../utils/ground_truth/headbox_wearer_speaker/' + fn + '_w.txt', delimiter=' ')
    res = res.astype('int')
    os.system('mkdir train_test_data/' + str(fileno))
    for frame_num in range(9000):
        print(fileno, frame_num)
        a = int(frame_num / 30 * 1000 / 5) 
        b = int(a + 64)
        if (a >= spec.shape[1]) or (b >= spec.shape[1]):
           continue
        spec_crop = spec[:,a:b].copy()
        dat = {}
        dat['spec'] = spec_crop
        if frame_num in res:
            dat['label'] = 1
        else:
            dat['label'] = 0

        with open('train_test_data/' + str(fileno) + '/' + str(frame_num) + '.pickle', 'wb') as handle:
            pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        
