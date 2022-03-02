#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 extract_audios.py 

import os

with open('voxceleb2_videos.txt') as f:
    content = f.readlines()
    video_names = [x.strip() for x in content]

pid_count = {}
for vn in range(len(video_names)):
    print(vn)
    vdir = video_names[vn].split('/')
    if not os.path.isdir('audios/' + vdir[2]):
         os.system('mkdir audios/' + vdir[2])

    
    if vdir[2] not in pid_count:
       pid_count[vdir[2]] = 1

    cmd =  'ffmpeg -i ' + video_names[vn] + ' -vn -acodec copy audios/' + vdir[2] + '/' + str(pid_count[vdir[2]]) + '.mp4'
    pid_count[vdir[2]] = pid_count[vdir[2]] + 1
    os.system(cmd)

            
    

