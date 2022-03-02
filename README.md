# EGO4D Audio Visual Diarization Benchmark

The Audio-Visual Diarization (AVD) benchmark corresponds to characterizing _low-level_ information about conversational scenarios in the [EGO4D](https://ego4d-data.org/docs/) dataset.  This includes tasks focused on detection, tracking, segmentation of speakers and transcirption of speech content. To that end, we are proposing 4 tasks in this benchmark. 

For more information on Ego4D or to download the dataset, read: [Start Here](https://ego4d-data.org/docs/start-here/).

Overall >750 hours of conversational data is provided in the first version of the AVD dataset. Out of this approximately 50 hours of data has been annotated to support these tasks. This corresponds to 572 clips. Of these 389 are training, 50 are validation and the remaining will are used for testing. 
Each clip is 5 minutes long.  The following schema summarizes some data statistics of the clips.
Speakers per clip : 4.71  
Speakers per frame : 0.74  
Speaking time in clip : 219.81 sec  
Speaking time per person in clip : 43.29 sec  
Camera wearer speaking time : 77.64 sec

**[Localization & Tracking](./tracking/README.md)** :  The goal of this task is to detect all the speakers in the visual field of view and track them in the video clip. 
We provide bounding boxes for each participant's face to enable this task.

**[Active speaker detection](./active-speaker-detection/)** :  In this task each of the tracked speakers are assigned an anonymous label, 
including the camera wearer who never appears in the visual field of view. 

**Diarization** ([**Audio Only**](./diarization/audio-only/README.md) or [**Audio Visual**](./diarization/audio-visual/README.md)) :  This task focuses on the voice activities of speakers who were localized, tracked and assigned anonymous labels from the previous 2 tasks. 
For this task, we provide the time segments corresponding to each speaker's voice activity in the clip.

**[Transcription](./transcription/README.md)** : 
For the last task, we transcribe the speech content. 

Please refer to this link for detailed annotations schema. 
https://ego4d-data.org/docs/benchmarks/av-diarization/#annotation-schema



