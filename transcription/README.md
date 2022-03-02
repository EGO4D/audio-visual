# README for the transcription task
Created on: Dec 10, 2021
Last update : Feb 22, 2021

Requirements:
- Install ffmpeg: https://ffmpeg.org/download.html
- Install sclite: https://github.com/usnistgov/SCTK.git
  - Add the bin folder to your PATH or adjust the PATH in score_asr.sh
- Download Kaldi English GLM: https://github.com/kaldi-asr/kaldi/blob/master/egs/ami/s5/local/english.glm
- Install PyTorch and espnet_model_zoo: https://github.com/espnet/espnet_model_zoo
- Install Python soundfile library (pip install soundfile) 
  
Assumptions:
- Assuming that you have already dowloaded the videos and annotations
- In this version, we will only use oracle segmentation for decoding and scoring ASR for the validation subset. 
  Hence, we need the av_val.json annotation files. Please check the notes section below for further details.

Running ASR:
1) Extract 16kHz single channel audio files in wav format from videos (if not done already)
   ./extract_wav.sh <video-dir> <output-wav-dir>
2) Modify the paths in the ./score_asr.sh to point to your ego4d clip directories and paths. 
   If you only want to decode a certain subset of the data (e.g. val or test), modify the "for" loops, accordingly. 
3) Extract transcriptions from the annotation files, decode audio and score the decoding output by running
   ./score_asr.sh <result-dir> 0
   (Note: If you have successfully ran stage 0 of the script (extract audio) and want the first and second stages, use 1 as the argument)


Model Description:
For this task, we use the pretrained "Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave" model from the espnet-model-zoo. This model was trained on Gigaspeech dataset. For the details of the model architecture, please refer to https://zenodo.org/record/4630406#.YhU5DJNKja4


Additional Notes:
- While running the decoding script for the first time, you may also need Python NLTK library.
Please follow the directions in the error message to install necessary subcomponents.

- System is tested on a Python 3.8.10 environment with the requirements_38_10.txt file.

- Current repo assumes that the oracle segmentation is available in the json files. However, in the real test condition, the user will not have access to the oracle segmentation information. In that case, the scripts can only be used to evaluate the val split. 

- For sclite scoring, we used the transcription (trn) format. 

- The current version of the decode_audio_oracle.py script contains a function called transcribe_segmented() which can generate a string by concatenating the ASR outputs of a given speech signal for a given window length, e.g. 5 seconds. However, this method requires that the reference transcription is also in this format, i.e. utterance-level transcription, which is hard to obtain from the segmented reference transcriptions due to overlapping speech regions. Using the segment time mark (STM) format would also not be a solution as we have time overlaps between reference segments. Overlapping segments show that we have multiple hypotheses for the same interval which cannot be handled by sclite.
