# Extract clipnames and video indices for training, testing and validation
# Usage: python3 extract_clipnames_and_split_indices.py 

import json
import os

def write_list(file_name, alist):
    with open(file_name, 'w') as f:
      for item in alist:
          f.write("%s\n" % str(item))


with open('av_train.json') as f:
  data = json.load(f)

clip_uids = []
clip_names = []
for n in range(len(data['videos'])):
   for m in range(len(data['videos'][n]['clips'])):
       clip_uid = data['videos'][n]['clips'][m]['clip_uid']
       clip_uids.append(clip_uid)
       clip_names.append(clip_uid + '.mp4')

with open('av_val.json') as f:
  data = json.load(f)

for n in range(len(data['videos'])):
   for m in range(len(data['videos'][n]['clips'])):
       clip_uid = data['videos'][n]['clips'][m]['clip_uid']
       clip_uids.append(clip_uid)
       clip_names.append(clip_uid + '.mp4')

#with open('av_test.json') as f:
#  data = json.load(f)
#
#for n in range(len(data['videos'])):
#   for m in range(len(data['videos'][n]['clips'])):
#       clip_uid = data['videos'][n]['clips'][m]['clip_uid']
#       clip_uids.append(clip_uid)
#       clip_names.append(clip_uid + '.mp4')

write_list('v.txt', clip_names)

with open('av_val.json') as f:
  data = json.load(f)

val_clip_indices = []
for n in range(len(data['videos'])):
   for m in range(len(data['videos'][n]['clips'])):
       val_clip_uid = data['videos'][n]['clips'][m]['clip_uid']
       val_clip_indices.append(clip_uids.index(val_clip_uid))

write_list('val.txt', val_clip_indices)

#with open('av_test.json') as f:
#  data = json.load(f)
#
#test_clip_indices = []
#for n in range(len(data['videos'])):
#   for m in range(len(data['videos'][n]['clips'])):
#       test_clip_uid = data['videos'][n]['clips'][m]['clip_uid']
#       test_clip_indices.append(clip_uids.index(test_clip_uid))
#
#write_list('test.txt', test_clip_indices)

with open('av_train.json') as f:
  data = json.load(f)

train_clip_indices = []
for n in range(len(data['videos'])):
   for m in range(len(data['videos'][n]['clips'])):
       train_clip_uid = data['videos'][n]['clips'][m]['clip_uid']
       train_clip_indices.append(clip_uids.index(train_clip_uid))
write_list('train.txt', train_clip_indices)
       
       
os.system('cp v.txt ../../active-speaker-detection/vad/v.txt')
os.system('cp v.txt ../../diarization/audio-visual/DER/der_train_on_ego4d/v.txt')
os.system('cp v.txt ../../diarization/audio-visual/DER/der_notrain_on_ego4d/v.txt')
os.system('cp v.txt ../../active-speaker-detection/wearer/energy_based/v.txt')
os.system('cp v.txt ../../active-speaker-detection/wearer/spectrogram/data/v.txt')
os.system('cp v.txt ../../active-speaker-detection/active_speaker/mrc_active_speaker_detection/prediction/v.txt')
os.system('cp v.txt ../../active-speaker-detection/active_speaker/active_speaker_evaluation/mAP_talknet/input/v.txt')
os.system('cp v.txt ../../active-speaker-detection/active_speaker/active_speaker_evaluation/mAP_mrc/input/v.txt')
os.system('cp v.txt ../../active-speaker-detection/audio_embedding/make_audio_embeddings/v.txt')
os.system('cp v.txt ../../active-speaker-detection/surrounding_people_audio_matching/mrc/v.txt')
os.system('cp v.txt ../../active-speaker-detection/surrounding_people_audio_matching/talknet/v.txt')
os.system('cp v.txt ../../tracking/v.txt')
os.system('cp v.txt ../../tracking/global_tracking/v.txt')

#os.system('cp test.txt ../../diarization/audio-visual/DER/der_train_on_ego4d/test.txt')
#os.system('cp test.txt ../../diarization/audio-visual/DER/der_notrain_on_ego4d/test.txt')
#os.system('cp test.txt ../../wearer/energy_based/test.txt')
#os.system('cp test.txt ../../active-speaker-detection/active_speaker/mrc_active_speaker_detection/prediction/test.txt')
#os.system('cp test.txt ../../active-speaker-detection/active_speaker/active_speaker_evaluation/mAP_talknet/input/test.txt')
#os.system('cp test.txt ../../active-speaker-detection/active_speaker/active_speaker_evaluation/mAP_mrc/input/test.txt')
#os.system('cp test.txt ../../active-speaker-detection/audio_embedding/make_audio_embeddings/test.txt')
#os.system('cp test.txt ../../active-speaker-detection/surrounding_people_audio_matching/mrc/test.txt')
#os.system('cp test.txt ../../active-speaker-detection/surrounding_people_audio_matching/talknet/test.txt')
#os.system('cp test.txt ../../tracking/test.txt')
#os.system('cp test.txt ../../active-speaker-detection/wearer/spectrogram/prediction')

os.system('cp val.txt ../../diarization/audio-visual/DER/der_train_on_ego4d/val.txt')
os.system('cp val.txt ../../diarization/audio-visual/DER/der_notrain_on_ego4d/val.txt')
os.system('cp val.txt ../../active-speaker-detection/wearer/energy_based/val.txt')
os.system('cp val.txt ../../active-speaker-detection/active_speaker/mrc_active_speaker_detection/prediction/val.txt')
os.system('cp val.txt ../../active-speaker-detection/active_speaker/active_speaker_evaluation/mAP_talknet/input/val.txt')
os.system('cp val.txt ../../active-speaker-detection/active_speaker/active_speaker_evaluation/mAP_mrc/input/val.txt')
os.system('cp val.txt ../../active-speaker-detection/audio_embedding/make_audio_embeddings/val.txt')
os.system('cp val.txt ../../active-speaker-detection/surrounding_people_audio_matching/mrc/val.txt')
os.system('cp val.txt ../../active-speaker-detection/surrounding_people_audio_matching/talknet/val.txt')
os.system('cp val.txt ../../tracking/val.txt')
os.system('cp val.txt ../../active-speaker-detection/wearer/spectrogram/prediction')

os.system('cp train.txt ../../active-speaker-detection/wearer/spectrogram/train/train.txt')

