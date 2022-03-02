#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 single_run.py directory_of_ego4d_videos video_index 

import os
import numpy as np
import sys

video_dir = sys.argv[1]

with open('v.txt') as file:
    lines = file.readlines()
    video_files = [line.rstrip() for line in lines]

video_id = int(sys.argv[2]) 

os.system('cd people_detection; rm -rf results; mkdir results; python3 detect_head_boxes.py ' + video_dir + '/' + video_files[video_id])
os.system('cd short_term_tracking/build; ./short_term_tracker ' + video_dir + '/' + video_files[video_id] + ' ../../people_detection')
os.system('cd global_tracking; python3 make_trajectories.py ' + video_dir + ' ' + str(video_id) + '; python3 group_fast.py; python3 save_results_ascii.py ' + video_dir + ' ' + str(video_id)) 

os.system('python3 show_tracking_result.py ' + video_dir + ' ' + str(video_id)) 
