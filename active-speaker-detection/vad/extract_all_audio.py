#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python3 extract_all_audio.py directory_of_ego4d_videos 

import numpy as np
import os
import sys

video_dir = sys.argv[1]

with open('v.txt') as f:
    content = f.readlines()
filenames = [x.strip() for x in content]

for k in range(len(filenames)):
    if not os.path.isfile(video_dir + '/' + filenames[k]):
       continue	    
    os.system('ffmpeg -i ' + video_dir + '/' + filenames[k] + ' -acodec pcm_s16le -ac 1 -ar 16000 audios/' + str(k) + '.wav')
    


