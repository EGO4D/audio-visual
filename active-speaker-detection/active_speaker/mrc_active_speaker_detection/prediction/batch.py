#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Summary: Detect active speakers using mouth region classification models

import os
import sys
import numpy as np

video_dir = sys.argv[1] 
tracking_results_dir = '../../../../tracking/global_tracking/results'
model_train = sys.argv[2] # model_train: 'ego4d' or 'ava'
data_set = sys.argv[3] # 'test' or 'val' 

if not os.path.isfile(data_set + '.txt'):
   sys.exit(data_set + ' list does not exist.')

video_nums = np.loadtxt(data_set + '.txt')
video_nums = video_nums.astype('int').tolist()

for video_id in video_nums:
    print('build/mrc ' + video_dir + ' ' + tracking_results_dir + ' ' + str(video_id) + ' ../../../../models/speaker_' + model_train + '.pt')
    os.system('build/mrc ' + video_dir + ' ' + tracking_results_dir + ' ' + str(video_id) + ' ../../../../models/speaker_' + model_train + '.pt')
    os.system('mv result.txt results/' + str(video_id) + '.txt')
