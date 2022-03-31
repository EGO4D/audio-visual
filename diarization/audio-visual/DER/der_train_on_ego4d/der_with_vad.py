#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 der_with_vad.py [test|val] 

import pickle
import numpy as np
import sys
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import os
import statistics

def get_intervals(t):
    intervals = []
    a = 0
    b = 0
    while (a < t.shape[0] and b < t.shape[0]):
        if (t[a] == 0 and t[b] == 0):
            a = a + 1
            b = b + 1
            continue
    
        if (t[a] == 1 and t[b] == 1):
            b = b + 1
            continue
    
        if (t[a] == 1 and t[b] == 0):
            intervals.append([a, b-1])
            a = b
            continue
    
    if (a != t.shape[0]):
        intervals.append([a, t.shape[0]-1])

    return intervals    

surrounding_people_audio_matching_results_dir = '../../../../active-speaker-detection/surrounding_people_audio_matching/results'
vad_results_dir = '../../../../active-speaker-detection/vad/vads'
wearer_audio_matching_results_dir = '../../../../active-speaker-detection/wearer/results'
gt_dir = '../../../../utils/ground_truth/headbox_wearer_speaker'

with open('v.txt') as f:
    content = f.readlines()
filenames = [x.strip() for x in content]

data_set = sys.argv[1] # 'test' or 'val'
if data_set == 'test':
   sys.exit('No test set annotations.')

video_nums = np.loadtxt(data_set + '.txt')
video_nums = video_nums.astype('int').tolist()

ders = []
for file_no in video_nums:
    fn = filenames[file_no].split('.')[0]
    with open(gt_dir + '/'  + str(fn) + '_s.pickle', 'rb') as handle:
        speech_truth = pickle.load(handle)
    
    trj_truth = {}
    for frame_num in speech_truth:
        for pid in speech_truth[frame_num]:
            if pid not in trj_truth:
               trj_truth[pid] = []
            trj_truth[pid].append(frame_num)
    
    vad = np.loadtxt(vad_results_dir + '/' + str(file_no) + '.txt')
    
    res = np.loadtxt(surrounding_people_audio_matching_results_dir + '/' + str(file_no) + '.txt')
    if len(res.shape) == 1:
       res = np.expand_dims(res, 1)

    tlen = np.loadtxt(surrounding_people_audio_matching_results_dir + '/t' + str(file_no) + '.txt')
    if len(tlen.shape) == 1:
       tlen = np.expand_dims(tlen, 0)
       
    wres = np.loadtxt(wearer_audio_matching_results_dir + '/' + str(file_no) + '.txt')

    hypothesis = Annotation()
    for k in range(res.shape[1]):  
        if (len(tlen) > k) and (len(tlen[k]) >= 2 ) and (tlen[k][1] < 100):
            continue
        tres = res[:, k]
        tres = ((tres < 0.225) + 0)
        for m in range(tres.shape[0]):
            if int(m/30.0*16000.0) < vad.shape[0]:
               tres[m] = tres[m] * vad[int(m/30.0*16000.0)]
    
        tres_intervals = get_intervals(tres)
        for n in range(len(tres_intervals)):
            hypothesis[Segment(tres_intervals[n][0], tres_intervals[n][1])] = str(k)
    
    wtres = ((wres > 0.35) + 0)
    for m in range(wtres.shape[0]):
        if int(m/30.0*16000.0) < vad.shape[0]:
           wtres[m] = wtres[m] * vad[int(m/30.0*16000.0)]
    
    wtres_intervals = get_intervals(wtres)
    for n in range(len(wtres_intervals)):
        hypothesis[Segment(wtres_intervals[n][0], wtres_intervals[n][1])] = str(-1)
    
    reference = Annotation()
    for k in trj_truth:
        if k < 0:
           continue
        g = np.zeros(9000).astype('int')
        for n in range(9000):
            if (n in trj_truth[k]):
               g[n] = 1 
    
        g_intervals = get_intervals(g)
        for n in range(len(g_intervals)):
            reference[Segment(g_intervals[n][0], g_intervals[n][1])] = str(k)
    
    diarizationErrorRate = DiarizationErrorRate()
    der = diarizationErrorRate(reference, hypothesis, uem=Segment(0, 9000))
    ders.append(der)
    print(file_no, der)

print('mean DER = ', statistics.mean(ders))    
