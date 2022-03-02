## Device wearer voice activity detection

We implemented two methods to detect the device wearer's
voice activity. The energy based method is in the folder energy_based.
The spectrogram classification method is in the folder spectrogram.

### Energy based method

In the command line:
```
cd energy_based
python3 short_time_energy.py <directory_of_ego4d_videos> <data_set>
python3 match_wearer_audio.py <data_set>
```
Make sure the directory_of_ego4d_videos points to the ego4d videos and
data_set is either test or val. The result is saved in the ../results 
directory. Each value in the output indicates the distance of the voice
at that instant to the device wearer's voice.

### Spectrogram classification
#### Prepare data
Before running the training and prediction code, run the following scripts
in the data directory (install octave first if you do not have one):
```
cd spectrogram/data
octave-cli batch_spectrogram.m
python3 make_train_test_data.py 
```
#### Prediction
A prediction model has been pre-trained. In spectrogram/prediction, run 
```
python3 predict.py [test|val] 
```
to predict the wearer voice probability for all the videos in the test 
or validation dataset.

#### Training
The training code is included in the directory spectrogram/train.      


 
