#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 match_audio.py asd_results_dir [test|val]

import pickle
import sys
import os
import numpy as np
from scipy.spatial.distance import cdist

asd_results_dir = sys.argv[1]
data_set = sys.argv[2] # 'test' or 'val'

if not os.path.isfile(data_set + '.txt'):
   sys.exit(data_set + ' list does not exist.')

video_nums = np.loadtxt(data_set + '.txt')
video_nums = video_nums.astype('int').tolist()

for fileno in video_nums:
    print(fileno, end = '\r')
    res = np.loadtxt(asd_results_dir + '/' + str(fileno) + '.txt')
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)
    
    box = {};
    
    for n in range(res.shape[0]):
        if not (int(res[n][1]) in box):
           box[int(res[n][1])] = {}
        box[int(res[n][1])][int(res[n][0])] = [int(res[n][2]), int(res[n][3]), int(res[n][4]), int(res[n][5]), int(res[n][6]), res[n][7]];
    
    with open('../../audio_embedding/make_audio_embeddings/embeddings/' + str(fileno) + '.pickle', 'rb') as f:
            audio_feature = pickle.load(f)
    
    afeat = {}
    for k in box:
        afeat[k] = []
        for frame_num in box[k]:
            p = box[k][frame_num]
            if p[5] > 0.99 and frame_num < len(audio_feature): 
                  afeat[k].append(audio_feature[frame_num])
    
    traj_len = []
    for k in afeat:
           traj_len.append([k, len(box[k])])
        
    np.savetxt('../results/t' + str(fileno) + '.txt', traj_len, fmt='%d')
   
    dists = []    
    for frame_num in range(0,min(9000,len(audio_feature))):
        dist = []
        ss = []
        q = audio_feature[frame_num]
        q = np.expand_dims(np.array(q), 0)
        for k in afeat:
            if len(afeat[k]) > 0:
               f = np.array(afeat[k])
               d = cdist(q, f)
               d = np.min(d)
               dist.append(d)
            else:
               dist.append(1000) 
        dists.append(dist)
   
    np.savetxt('../results/' + str(fileno) + '.txt', dists, fmt='%f')
   
