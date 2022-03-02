# Format tracking ground truth to the MOT format
# Usage: python3 make_mot_ground_truth.py <video_dir> [test | val]

import os
import sys
import cv2 as cv
import numpy as np

video_dir = sys.argv[1]
data_set = sys.argv[2] 
if data_set == 'test':
   sys.exit('No test annotations.')

os.system('cp ' + data_set + '.txt tracking_evaluation/mot_challenge/seqmaps/mytrack-test.txt')
os.system("sed -i '1 i\\name' tracking_evaluation/mot_challenge/seqmaps/mytrack-test.txt")

with open('v.txt') as f:
    content = f.readlines()
filenames = [x.strip() for x in content]

for fileno in range(len(filenames)): 
    print(fileno)
    fn = (filenames[fileno]).split('.')[0]
    res = np.loadtxt('headbox_wearer_speaker/' + fn + '.txt', delimiter=' ')
    res = res.astype('int')
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)
    
    mot_res = []
    ids = {}
    for n in range(res.shape[0]):
        if int(res[n][0]) not in ids:
            ids[int(res[n][0])] = []
        if int(res[n][1]) not in ids[int(res[n][0])]:
           mot_res.append([res[n][0]+1, res[n][1], res[n][2], res[n][3], res[n][4]-res[n][2], res[n][5]-res[n][3], -1, -1, -1, -1])
           ids[int(res[n][0])].append(int(res[n][1]))
    
    os.system('mkdir tracking_evaluation/mot_challenge/mytrack-test/' + str(fileno))
    os.system('mkdir tracking_evaluation/mot_challenge/mytrack-test/' + str(fileno) + '/gt')
    np.savetxt('tracking_evaluation/mot_challenge/mytrack-test/' + str(fileno) + '/gt/gt.txt', mot_res, fmt='%d', delimiter=',')

    cap = cv.VideoCapture(video_dir + '/' + filenames[fileno])
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) + 1
    fps = cap.get(cv.CAP_PROP_FPS)
    ret, frame = cap.read()
    with open('tracking_evaluation/mot_challenge/mytrack-test/' + str(fileno) + '/seqinfo.ini', 'w') as f:
        f.write('[Sequence]\n')
        f.write('name=' + str(fileno) + '\n')
        f.write('imDir=img1\n')
        f.write('frameRate=' + str(fps) + '\n')
        f.write('seqLength=' + str(total) + '\n')
        f.write('imWidth=' + str(frame.shape[1]) + '\n')
        f.write('imHeight=' + str(frame.shape[0]) + '\n')
        f.write('imExt=.jpg\n')
    cap.release()    
  
