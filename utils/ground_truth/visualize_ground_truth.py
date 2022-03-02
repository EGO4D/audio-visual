# Visualize the ground truth annotations
# Usage: python3 visualize_ground_truth.py <video_dir> <clip_uid>

import cv2 as cv
import numpy as np
import pickle
import sys

video_dir = sys.argv[1]
clip_uid = sys.argv[2]

with open('headbox_wearer_speaker/' + clip_uid + '_s.pickle', 'rb') as handle:
    voice = pickle.load(handle)

res = np.loadtxt('headbox_wearer_speaker/' + clip_uid + '.txt') 
if len(res.shape) == 1:
    res = np.expand_dims(res, axis=0)

box = {};

for n in range(res.shape[0]):
    if not (res[n][0] in box):
       box[res[n][0]] = []       
    box[res[n][0]].append([res[n][1], res[n][2], res[n][3], res[n][4], res[n][5], res[n][6]]);

fname = video_dir + '/'  + clip_uid + '.mp4'

cap = cv.VideoCapture(fname)
frame_num = 0
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    if frame_num in voice:
       wearer_speak = (0 in voice[frame_num])

       if wearer_speak:
          cv.putText(frame, 'Wearer speaking', (100,100), cv.FONT_HERSHEY_SIMPLEX,
                 2, (255,0,255), 5, cv.LINE_AA)

       all_talker = ''
       for talker in voice[frame_num]:
           all_talker = all_talker + ' ' + str(talker)

       if all_talker:
          cv.putText(frame, all_talker + ' speaking', (100,150), cv.FONT_HERSHEY_SIMPLEX,
                           2, (255,0,0), 5, cv.LINE_AA)
           
    if frame_num in box:  
       b = box[frame_num]
       for k in range(len(b)):
           pid = int(b[k][0])
           x1 = int(b[k][1])
           y1 = int(b[k][2])
           x2 = int(b[k][3])
           y2 = int(b[k][4])
           speak = int(b[k][5]);
           
           if (speak == 0):
               cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 5)
           else:
               cv.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 50)
               
           font = cv.FONT_HERSHEY_SIMPLEX 
           fontScale = 2
           color = (255, 0, 0) 
           thickness = 5           
           cv.putText(frame, str(pid), ((x1+x2)//2,(y1+y2)//2), font,  
                   fontScale, color, thickness, cv.LINE_AA) 

    frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))          
    cv.imshow('Ground Truth', frame)
    cv.waitKey(5)
    frame_num = frame_num + 1

