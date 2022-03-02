#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 match_wearer_audio.py data_set 

import pickle
import sys
import os
import numpy as np
from scipy.spatial.distance import cdist

data_set = sys.argv[1] # 'test' or 'val'
if not os.path.isfile(data_set + '.txt'):
   sys.exit(data_set + ' list does not exist.')


video_nums = np.loadtxt(data_set + '.txt')
video_nums = video_nums.astype('int').tolist()

for vid in video_nums:
    aud_mag = np.loadtxt('energy/' + str(vid) + '.txt')
    aud_mag_max = np.max(aud_mag)
    aud_mag_min = np.min(aud_mag)
    aud_mag_th = (aud_mag_max * 0.2 + aud_mag_min * 0.8)
    
    with open('../../audio_embedding/make_audio_embeddings/embeddings/' + str(vid) + '.pickle', 'rb') as f:
            audio_feature = pickle.load(f)
    
    afeat = []
    for frame_num in range(9000):
            if frame_num < len(aud_mag) and aud_mag[frame_num] > aud_mag_th and frame_num < len(audio_feature): 
                  afeat.append(audio_feature[frame_num])
    dists = [] 
    for frame_num in range(min(9000,len(audio_feature))):    
        q = audio_feature[frame_num]
        q = np.expand_dims(np.array(q), 0)
        if len(afeat) > 0:
           f = np.array(afeat)
           d = cdist(q, f)
           d = np.min(d)
           dist = d
        else:
           dist = 1000 
    
        dists.append(dist)

    np.savetxt('../results/' + str(vid) + '.txt', dists, fmt='%f')
    
