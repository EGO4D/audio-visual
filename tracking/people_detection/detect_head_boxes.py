#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import darknet as dn
import cv2 as cv
import numpy as np
import timeit

net = dn.load_net(b'cfg/head_body.cfg', b'../../models/head_body.model', 0)
meta = dn.load_meta(b'cfg/head_body.data')

count = 0
cap = cv.VideoCapture(sys.argv[1])
while(1):
    ret, frame = cap.read()
    if frame is None:
       break
    frame = cv.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

    frame_org = frame.copy()
    cv.imwrite('test.jpg', frame);   
    start_time = timeit.default_timer()
    r = dn.detect(net, meta, bytes('test.jpg', encoding='utf-8'))
    timepassed  = timeit.default_timer()-start_time

    boxes = [];
    for k in range(len(r)):
        print(r[k][0], r[k][1], r[k][2])
        if r[k][0] != b'head':
            continue;
        x = r[k][2][0]
        y = r[k][2][1]
        w = r[k][2][2]
        h = r[k][2][3]
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        boxes.append([x1, y1, x2, y2, r[k][1]]);
        cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    
    cv.imshow('image', frame) 
    if len(boxes) > 0:
       np.savetxt('results/' + str(count) + '.txt', boxes, fmt='%d %d %d %d %f')
    else:
       np.savetxt('results/' + str(count) + '.txt', boxes)
 
    cv.waitKey(1)
    count = count + 1

cap.release()

