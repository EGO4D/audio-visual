## Global People Tracking

A set of tools to perform long term people tracking in videos.

### Installation

The following instructions are for Ubuntu 18.04. Newer versions of the Ubuntu should also work.

#### Dependencies
- cuda-toolkits
    - Tested on cuda-10.2. Newer versions may also work.
- python3, pip3
- Pytorch and libtorch
    - Tested on pytorch and libtorch 1.7.1. Newer versions should work. Make sure the pytorch and libtorch versions match the cuda version.
- cmake
- opencv-3.4.11
     - Installation from source using cmake is preferred. Make sure to install numpy (pip3 install numpy) before cmake. Opencv includes a copy of the ffmpeg in the source code. So no need to install ffmpeg separately. However, make sure you install the video codecs before cmake: sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev; sudo apt-get install libxvidcore-dev libx264-dev. You also need to install gtk, for instance, sudo apt install libgtk2.0-dev. Enable ffmpeg and disable cuda when generating the Makefile using cmake (mkdir build; cd build; cmake ../ -DWITH_FFMPEG=ON -DWITH_CUDA=OFF). 


#### People detection

People detection code is adapted from the darknet (https://github.com/pjreddie/darknet).  
Double check the directories in the people_detection/Makefile and make sure they match your system settings. 
Then in the command line.
```
cd people_detection
make -j
```
#### short_term_tracking
To build, 
```
cd short_term_tracking
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=your_libtorch_library_directory ..
make
```
### Run global people tracking
It is a good idea to test one video to verify the code runs correctly. To process one video in the ego4d dataset:
```
python3 single_run.py your_ego4d_videos_directory video_index
```
This will process the video at row video_index (row index starting from 0) in the v.txt file. The code runs in several rounds: 
the first detects the head boxes in videos, the second performs short term tracking and the third performs long term global tracking. 
The final result is stored in global_tracking/results.

As an example, try
```
python3 single_run.py your_ego4d_videos_directory 22 
```
To view the results again
```
python3 show_tracking_result.py your_ego4d_videos_directory 22
```

To process all the videos in the test or the validation set.
```
python3 batch.py your_ego4d_videos_directory [test|val]

```
The results will be dumped in the global_tracking/results. The dumped file name is the video id (video names are defined in v.txt). Each line has the format:
```
frame_num, person_id, head_box_left_upper_col, head_box_left_upper_row, head_box_right_bottom_col, head_box_right_bottom_row
```
### Evaluation: MOT metrics
To evaluate the tracking results, follow the steps below.
 
In the tracking_evaluation sub-directory,
```
git clone https://github.com/JonathonLuiten/TrackEval.git
mv data TrackEval
mv evaluate_global_tracker.sh TrackEval
```

We need to convert our tracking results to a format that is compatible with TrackEval.
```
cd format_tracking_results
python3 track2motformat.py
mv mot_format/*.txt ../TrackEval/data/trackers/mot_challenge/mytrack-test/global/data
```
Now, double check the mytrack-test.txt in TrackEval/data/gt/mot_challenge/seqmaps so that the video indices listed are consistent with the video tracking result list. (The list should usually be either the test video list or the validation video list.)

Finally,
```
cd ../TrackEval
bash evaluate_global_tracker.sh
```
