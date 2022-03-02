## Ego4d audio-visual diarization baseline

Baseline code for ego4d audio-visual diarization.
Here are the steps to install and run the code. 

### Download deep models
Download the model tar ball. 
```
python -m ego4d.cli.cli -y --output_directory /path/to/output/ --datasets av_models
```
Then, untar the file in the code's
root directory (ego4d_av_diaraization). The directory  
should be named as 'models'.
Inside the models directory, expand the snakers4_silero-vad_master.zip 
and then move the directory of 'snakers4_silero-vad_master'
to '~/.cache/torch/hub/'. 

### Preprocess ground truth data 
Download the ego4d videos and place them in a directory. Download
the AV ground truth annotations and move the json files to the utils/ground_truth directory. 
Then, sequentially run 
the following scripts in the directory ground_truth:

```
bash init_dirs.sh
python3 extract_clipnames_and_split_indices.py
python3 extract_boxes_and_speakers.py
python3 make_mot_ground_truth.py <video_dir> [test | val]
mv tracking_evaluation/mot_challenge ../../tracking/tracking_evaluation/data/gt
``` 
To visualize the audio-visual ground truth annotations, use the script visualize_ground_truth.py.

### People tracking
Follow [people_tracking/README.md](../../tracking/README.md) 
to compile the C++
code and run the python scripts. After completing this step, the tracking
results for either the test videos or the validation videos will be
saved in the global_tracking/results directory.

### VAD
Follow [vad/README.md](../../active-speaker-detection/vad/README.md) to compute the voice activity detection
results for all the videos in the ego4d diarization dataset.

### Active speaker detection
Follow [active_speaker/READE.md](../../active-speaker-detection/active_speaker/README.md) to 
run the active speaker detection code. Two active speaker detectors are included: MRC and TalkNet.

For MRC, after compiling the code,
the batch script batch.py in /mrc_active_speaker_detection/prediction
can be used to process all the test or validate videos based on
the tracking results saved in global_tracking/results.

For TalkNet, first check out the code from github using the script 'check_out_talknet.sh'
in the sub-directory TalkNet_ASD and then follow the instructions to execute the code.
The tracking results in the global_tracking/results need to be copied
to proper folders; see the Ego4d-TalkNet instructions for details.

After active speaker detection, follow
the instructions in the active_speaker/READNE.md to compute the mAPs
in active_speaker_evaluation.

### Audio embedding
Follow the instructions in [audio_embedding/README.md](../../active-speaker-detection/audio_embedding/README.md) to extract audio
feature embedding for the ego4d videos.

### Wearer voice activity detection
Follow the instructions in [wearer/README.md](../../active-speaker-detection/wearer/README.txt) to classify wearer voice
activity using either the energy based method or the
spectrogram approach.

### Surrounding people audio matching
Follow the instructions in [surrounding_people_audio_matching/README.md](../../active-speaker-detection/surrounding_people_audio_matching/README.md)
to match audio embedding of each speaker across the whole video so that
invisible speaker's voice can be detected.

### DER
The code in the directory DER computes the diarization error (DER) metric based on the AV diarization
results from the previous steps.
