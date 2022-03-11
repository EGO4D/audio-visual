# Description of Audio-only Diarization Baseline System

For the audio-only diarization baseline system, an approach based on Bayesian HMM clustering of x-vector sequences [(VBx)](https://github.com/BUTSpeechFIT/VBx) has been utilized. The method requires speech activity regions and these were obtained using the ASpIRE Voice Activity Detection (VAD) model based on a time delay neural network (TDNN) with statistics pooling. It is available [here](https://kaldi-asr.org/models/m4) with the [Kaldi](https://github.com/kaldi-asr/kaldi) speech recognition toolkit. Although this VAD has been trained on slightly different data (telephone conversations), and thus does not provide the best possible results, it has been chosen for the baseline system  because of its general availability.

The speech activity regions are uniformly segmented (1.44s long subsegments every 0.24s) to obtain shorter segments and speaker embeddings
(so-called x-vectors) are extracted one per subsegment. The x-vectors are obtained with a ResNet101 extractor trained to produce speaker-discriminative embeddings. The input of the neural network are log Mel-filter bank features every 10ms, and given a segment of speech, the model outputs a single 256 dimensional vector that represents the whole segment. The information of the whole segment is aggregated with a statistical pooling layer which computes the mean and standard deviation of activations over the time domain. After the pooling layer, a linear transformation is used to reduce the dimensionality to 256. During training, the last layer has one unit per speaker identity in the training set.

The x-vectors are initially clustered to a few dozens of classes using agglomerative hierarchical clustering. This initial clustering is fed as initialization to a Bayesian hidden Markov model which estimates altogether the number of speakers in the recording as well as the assignment of x-vectors to the states. Each state in the model corresponds to one speaker and the probability of observing a particular x-vector in a particular state can be interpreted as the corresponding speaker producing the corresponding segment of speech. The most relevant hyperparameters of the model were fine-tuned to obtain the best DER performance on the Ego4D validation set. The training [recipe](https://github.com/phonexiaresearch/VBx-training-recipe) was published by Phonexia Research.


## STEPS TO REPRODUCE EGO4D AUDIO-ONLY DIARIZATION BASELINE ON VALIDATION SET

1. All the following are expected to be run from the `diarization/audio-only` directory.

2. Create Python3 virtual environment based on `requirements.txt` and activate it. E.g. for virtualenv you can use the following commands to install everything necessary:

    `python3 -m venv ~/env_ego4d/`

    `source ~/env_ego4d/bin/activate`

    `pip install -U pip`

    `pip install wheel`

    `pip install -r requirements.txt`

    `pip install </path/to/Ego4D repository>`

    Then use this environment for all Python runs from this recipe. 

3. Download data. If you only want to download a part of the videos (e.g. clips from the validation set), create a file with ids of the desired videos and use --video_uid_file parameter; if you wish to download all the data, just omit it:

    `python3 -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets clips annotations --video_uid_file data_preparation/clips_val.txt`

4. If you do not have FFMPEG on your system, install it from your distribution repository or from https://www.ffmpeg.org/download.html FFMPEG binary will be called from the Python script in the next step. This recipe was tested with FFMPEG version 4.2.4-1ubuntu0.1

5. Extract audio clips from downloaded videos - for validation set, the command would look like this:
 `python3 data_preparation/video2audio.py --video_dir ~/ego4d_data/v1/clips/ --output_dir ~/ego4d_data/v1/audio_clips/`

6. Generate reference rttms from the annotation json:
    `python3 data_preparation/generate_voice_rttms.py --json-filename ~/ego4d_data/v1/annotations/av_val.json --rttm-output-dir ~/ego4d_data/v1/rttms --lab-output-dir ~/ego4d_data/v1/vad_ref`

7. Download the KALDI toolkit from https://github.com/kaldi-asr/kaldi Read the instructions from the INSTALL file and build it.

8. Make sure that [sox](http://sox.sourceforge.net/) is installed on your system. If not, please install it.

9. In the script `vad_kaldi/run_vad.sh`, set all the variables (especially KALDI_ROOT) to respect your setup and run the script. The download of the ASpIRE VAD model from the Kaldi repository is included in the script.

10. Run python script vad_kaldi/seg2lab.py: `python3 vad_kaldi/seg2lab.py ~/ego4d_data/v1/vad_output_dir`

11. [THIS STEP IS OPTIONAL] To compare ref voice segments and Kaldi VAD, and to verify that all the files are matching, you can run: `python3 vad_kaldi/compare_vads.py --dir1 ~/ego4d_data/v1/vad_ref --dir2 ~/ego4d_data/v1/vad_output_dir`
The output for the validation should look like below (actual numbers may differ slightly due to a random dither applied at the feature extraction step):
     ```
     Analyzing 50 pairs of files
     Dir1 VAD sum: 6810.14 sec  Dir2 VAD sum: 5974.66 sec
     VAD intersection abs: 4757.81 sec
     VAD intersection rel Dir1: 69.86 %   VAD intersection rel Dir2: 79.63 %
     ```

12. Install VBx Diarization toolkit from https://github.com/BUTSpeechFIT/VBx

13. Set the variables inside the VBx diarization script and then call it. By changing the variable LAB_DIR you can control whether VAD segments used are those generated by Kaldi or ref generated from the annotation JSON. So following the directory structure from this recipe, you would first set EXP_DIR=\~/ego4d_data/v1/vbx_diarization_vad and LAB_DIR=\~/ego4d_data/v1/vad_output_dir, and then EXP_DIR=\~/ego4d_data/v1/vbx_diarization_ref and LAB_DIR=\~/ego4d_data/v1/vad_ref

     `diarization/run_vbx_diarization.sh`

     After the run, the diarization output (in [RTTM format](https://web.archive.org/web/20170119114252/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf)) will be in $EXP_DIR/out_dir_AHC+VB/rttms/ and the corresponding scores in $EXP_DIR/out_dir_AHC+VB/result_full

14. To compute Diarization Error Rate, you can also use the following script:

     `python3 diarization/score_pyannote.py --ref_dir ~/ego4d_data/v1/rttms/ --hyp_dir ~/ego4d_data/v1/vbx_diarization/out_dir_AHC+VB/rttms/`
    
     The numbers should be the same as in $EXP_DIR/out_dir_AHC+VB/result_full

15. If you wish to convert the output RTTM files to a single file in the JSON format convenient for Ego4D Audio-only Diarization Challenge, you can use the following script:

     `python3 diarization/rttms2json.py --rttm_dir ~/ego4d_data/v1/vbx_diarization/out_dir_AHC+VB/rttms/ --output ~/ego4d_data/v1/vbx_diarization/out_dir_AHC+VB/output.json`


