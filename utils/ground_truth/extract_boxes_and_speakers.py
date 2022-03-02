# Extract head boxes and voice activities from the annotation
# Usage: python3 extract_boxes_and_speakers.py 

import json
import numpy as np
import pickle

def extract_boxes_speakers(annotation_file):
    with open(annotation_file) as f:
      data = json.load(f)
    
    for n in range(len(data['videos'])):
        for m in range(len(data['videos'][n]['clips'])):
            print(n, m, data['videos'][n]['clips'][m]['clip_uid'])
            clip_uid = data['videos'][n]['clips'][m]['clip_uid']
            box = {}
            voice = {}
            wearer_speak = []
            for k in range(len(data['videos'][n]['clips'][m]['persons'])):
                pid = int(data['videos'][n]['clips'][m]['persons'][k]['person_id'])
                is_camera_wearer = data['videos'][n]['clips'][m]['persons'][k]['camera_wearer']    
                for vseg in data['videos'][n]['clips'][m]['persons'][k]['voice_segments']:
                    for frame_num in range(max(0, vseg['start_frame']), min(9000, vseg['end_frame']+1)):
                        if frame_num not in voice:
                            voice[frame_num] = []
                        voice[frame_num].append(pid)
                        if is_camera_wearer:
                           wearer_speak.append(frame_num)
            
                head_boxes = data['videos'][n]['clips'][m]['persons'][k]['tracking_paths']
                voice_segments = data['videos'][n]['clips'][m]['persons'][k]['voice_segments']
                for tracklet in head_boxes:
                    for t in tracklet['track']:
                        frame_num = t['frame']
                        x1 = int(t['x'])
                        y1 = int(t['y'])
                        x2 = int(x1 + t['width'])
                        y2 = int(y1 + t['height'])
                        if t['frame'] not in box:
                           box[frame_num] = []
                        box[frame_num].append([pid, x1, y1, x2, y2])
            
            np.savetxt('headbox_wearer_speaker/' + clip_uid + '_w.txt', wearer_speak, fmt='%d')
            
            output = []
            for frame_num in range(0, 9000):
                if frame_num in box:
                    for k in range(len(box[frame_num])): 
                        if (frame_num in voice) and (box[frame_num][k][0] in voice[frame_num]): 
                           output.append([frame_num] + box[frame_num][k] + [1])
                        else:
                           output.append([frame_num] + box[frame_num][k] + [0])                        
            
            np.savetxt('headbox_wearer_speaker/' + clip_uid + '.txt', output, fmt='%d %d %d %d %d %d %d')
            
            with open('headbox_wearer_speaker/' + clip_uid + '_s.pickle', 'wb') as handle:
                pickle.dump(voice, handle, protocol=pickle.HIGHEST_PROTOCOL)



extract_boxes_speakers('av_train.json')
extract_boxes_speakers('av_val.json')

