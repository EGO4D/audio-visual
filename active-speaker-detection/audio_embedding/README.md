## Generate voice embedding

### Voice embedding
To match audio to invisible speakers, we construct the voice feature
embeddings using
```
cd make_audio_embeddings
python3 batch_audio_embedding.py <directory_of_ego4d_videos> [test|val]
```
The pre-trained voice embedding model is included. If you 
want to retrain the model, the training instruction is as follows.

### Training
The model is trained on the voxceleb2 data set. The voxceleb2 videos
are defined in the voxceleb2_videos.txt. Make sure the videos are saved
to the proper directory before running the scripts. 
To speedup the data loading
we first extract all audio in mp4 format using
``` 
python3 extract_audios.py
```

To train the model:
```
python3 train.py
```
 
