## Surrounding people voice matching 

Based on the active speaker detection, we further match the voice of a 
"speaker" across the whole video so we can detect the voice activity 
even when the speaker is invisible. The voice matching is based on
the audio embedding.

Since the output formats of MRC and TalkNet are slightly different, we 
use two scripts to process their outputs. Voice matching based on MRC 
is audio_match.py in the sub-directory mrc and the code for voice matching based 
on talknet is audio_math.py in talknet. To run the code, go 
to the correspond sub-directory and run
```
python3 audio_match.py asd_results_dir [test|val]
```
asd_results_dir is the active speaker detection result directory for the MRC or TalkNet. 
audio_match.py will process all the videos in the test or validation dataset.

 

  


