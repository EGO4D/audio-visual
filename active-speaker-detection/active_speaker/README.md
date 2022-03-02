## Active Speaker Detection (ASD)

Code for detecting the active speakers in the visual field of view. Two methods are used. The first classifies speaking or non-speaking using the crop mouth region images. The second one is based on the TalkNet_ASD (https://github.com/TaoRuijie/TalkNet_ASD).

### Mouth region classification (MRC)
The code is implemented using C++ and libtorch. To compile the code:
```
cd mrc_active_speaker_detection/prediction
mkdir build
cd build
cmake  -DCMAKE_PREFIX_PATH=directory_of_libtorch ../
make

```
After compliation is complete, to verify the code is correctly compiled,
```
cd ..
python3 run_once.py <ego4d_video_directory> [ego4d|ava] <video_id>
```
By setting model to ego4d and video_id to 22, you should get almost perfect result because video 22 is from the training set. 
Also check
the result by setting model to be ava.  

Run the following batch process to predict active speakers in each video from the test or validation dataset. 
The argument 'ego4d' indicates using the model trained from ego4d train set, while 'ava' indicates we use the model 
trained from the Google AVA active speaker dataset. The argument 'test' or 'val' indicates whether we process all the 
videos in the test set or the validation set.
```
python3 batch.py direcotory_of_ego4d_videos [ego4d|ava] [test|val]
```
The results are dumped in the results directory.

## TalkNet_ASD
First, check out the code of TalkNet code adapted to the ego4d dataset using:
```
cd TalkNet_ASD
bash check_out_talknet.sh
```
Then, follow the TalkNet readme to prepare the data and run the training and prediction code.

## Active speaker detection evaluation
The code in active_speaker_evaluation/mAP_mrc and active_speaker_evaluation/mAP_talknet are used to compute the mAP scores of the mrc and talknet ressults separately. The mAP metric implementation is from https://github.com/Cartucho/mAP.

In active_speaker_evaluation/mAP_mrc/input and active_speaker_evaluation/mAP_talknet/input, run the command lines to prepare the data.
 ```
 python3 format_ground_truth.py [test|val] # format ground truth labeling in the test or validation dataset.
 ```
In active_speaker_evaluation/mAP_mrc/input
 ```
 python3 format_asd_with_smoothing.py asd_results_dir [test|val]
 ```
or
```
python3 format_asd_with_smoothing_vad.py asd_results_dir vad_results_dir [test|val]

```

The vad_results_dir is at ego4d_av_diarization/vad/vads.
For MRC the asd_results_dir is at ego4d_av_diarization/active_speaker/mrc_active_speaker_detection/prediction/results. 

The data formater for the talknet is slightly different. In active_speaker_evaluation/mAP_talknet/input
```
python3 format_asd.py asd_results_dir [test|val]
```
or
```
python3 format_asd_vad.py asd_results_dir vad_results_dir [test|val]
```
After data formating, go back to the mAP_mrc or mAP_talknet, run
```
python3 main.py
```
