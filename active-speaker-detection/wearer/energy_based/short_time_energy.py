#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 short_time_energy.py directory_of_ego4d_videos data_set

import numpy as np
import cv2 as cv
import os.path
import audiosegment
import sys
import pickle

video_dir = sys.argv[1]
data_set = sys.argv[2] # 'test' or 'val'

if not os.path.isfile(data_set + '.txt'):
   sys.exit(data_set + ' list does not exist.')

with open('v.txt') as file:
    lines = file.readlines()
    video_files = [line.rstrip() for line in lines]

video_nums = np.loadtxt(data_set + '.txt')
video_nums = video_nums.astype('int').tolist()

for video_id in video_nums:
     vname = video_dir + '/' + video_files[video_id]
     print(vname)
     cap = cv.VideoCapture(vname)
     video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
     video_fps =  cap.get(cv.CAP_PROP_FPS)
     cap.release()
     
     sound = audiosegment.from_file(vname)
     sound = sound.resample(sample_rate_Hz=16000, sample_width=2, channels=1)
     samples = sound.get_array_of_samples()
     fps = 16000
     aud = np.array(samples).astype('float32')/65536.0
     #total_time = int(aud.shape[0]/fps)
     
     eng = []
     for frame_num in range(video_frames):
         afirst = int(frame_num * 1.0 / float(video_fps) * fps - 0.5*fps)
         alast = int(afirst + fps)
     
         if afirst < 0:
            afirst = 0
        
         if alast >= aud.shape[0]:
            alast = aud.shape[0]-1 
         
         y = aud[afirst:alast]
     
         short_time_eng = np.mean(np.abs(y))
         eng.append(short_time_eng) 
        
     np.savetxt('energy/' + str(video_id) + '.txt', eng, fmt='%f') 




