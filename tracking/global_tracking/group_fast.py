#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import json
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
import pickle
import sys

sys.stdout.write('\033[K')

with open('traj.pkl', 'rb') as handle:
    feature = pickle.load(handle)

weights = {}
conflict = {}
key = list(feature.keys())
for n in key:
    print(n, end = '\r')
    f1 = np.array(feature[n])
    if len(f1.shape) == 1:
        f1 = np.expand_dims(f1, axis=0)
    f1_frames = f1[:,0]
    f1 = f1[:,5:]

    for m in key:
        if n == m:
            continue
        f2 = np.array(feature[m])
        if len(f2.shape) == 1:
             f2 = np.expand_dims(f2, axis=0)
        f2_frames = f2[:,0]
        f2 = f2[:,5:]

        d = 1000
        if np.intersect1d(f1_frames, f2_frames).size == 0:
           d = cdist(f1, f2)
           d = d.min() #(d.min() - 1)*(f1.shape[0]+f2.shape[0])
           conflict[(n,m)] = 0
           conflict[(m,n)] = 0           
        else:
           conflict[(n,m)] = 1
           conflict[(m,n)] = 1
            
        weights[(n,m)] = d
        weights[(m,n)] = d
    
g = {}
for k in feature:
    g[k] = [k]

while True:
      ww = []
      pp = []
      key = list(g.keys())
      K1 = len(key)

      if (K1 == 1):
         break 

      for n in key:
          for m in key:
              if n == m:
                  continue

              conf = 0
              minw = []
              for p in g[n]:
                  for q in g[m]:
                      if conflict[(p,q)] == 1:  
                         conf = 1
                      minw.append(weights[(p,q)])

              f1_len = 0
              for p in g[n]:
                  f1_len = f1_len + len(feature[p])

              f2_len = 0
              for q in g[m]:
                  f2_len = f2_len + len(feature[q])
                                                             
              if conf == 1:
                  d = 1000
              else:    
                  d = (np.min(minw) - 0.75) 
      
              ww.append(d)
              pp.append([n, m])
      
      
      k = np.argmin(ww)
      if ww[k] < 0:
         n = pp[k][0]
         m = pp[k][1]
         g[n] = g[n] + g[m]
         del g[m]
      
         print('**', K1, len(g.keys()), n, m, k, ww[k], end = '\r')   
      else:
         break 
    

gres = {}
for k in g:
    if k not in gres:
       gres[k] = []
    for p in g[k]:   
        gres[k] = gres[k] + list(np.array(feature[p])[:,0:5])

with open('result_group_fast.pkl', 'wb') as f:
      pickle.dump(gres, f, pickle.HIGHEST_PROTOCOL)
    

