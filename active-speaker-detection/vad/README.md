## Voice Activity Detection (VAD)

Python3 scripts to extract audio and detect voice activity.

### Usage
First, extract the audio in ego4d videos using ffmpeg. The sampling rate is 16kHz. The bit length per sample is 16-bit. The audio is single channel.

```
  python3 extract_all_audio.py directory_of_ego4d_videos
```
The audio files are saved in the 'audios' directory.

Then, compute the voice activities of each audio file.
```
 python3 vad.py
```
The VAD results are saved in the 'vads' directory.
