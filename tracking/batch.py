#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Summary: Batch process of people tracking and 
#          dumping the results.
# Usage: python3 batch.py directory_of_ego4d_videos [test|val] 

import os
import numpy as np
import sys

video_dir = sys.argv[1]
data_set = sys.argv[2]  # 'test' or 'val'

if not os.path.isfile(data_set + '.txt'):
   sys.exit(data_set + ' list does not exist.')  

with open('v.txt') as file:
    lines = file.readlines()
    video_files = [line.rstrip() for line in lines]

video_nums = np.loadtxt(data_set + '.txt')
video_nums = video_nums.astype('int').tolist()

for video_id in video_nums:
    os.system('cd people_detection; rm -rf results; mkdir results; python3 detect_head_boxes.py ' + video_dir + '/' + video_files[video_id])
    os.system('cd short_term_tracking/build; ./short_term_tracker ' + video_dir + '/' + video_files[video_id] + ' ../../people_detection')
    os.system('cd global_tracking; python3 make_trajectories.py ' + video_dir + ' ' + str(video_id) + '; python3 group_fast.py; python3 save_results_ascii.py ' + video_dir + ' ' + str(video_id)) 

