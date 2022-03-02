#!/usr/bin/bash
set -e

# Path to Kaldi installation root dir, please change according to your local setup 
KALDI_ROOT=~/repos/kaldi
# Path to dir with audio files (wavs)
AUDIO_CLIPS_DIR=~/ego4d_data/v1/audio_clips

# Directory where Kaldi data preparation files will be created
DATA_DIR=~/ego4d_data/v1/kaldi_data_dir
# Directory where Kaldi MFCCs will be stored 
MFCC_DIR=~/ego4d_data/v1/kaldi_mfcc_dir
# Output directory for VAD segments
VAD_OUTPUT_DIR=~/ego4d_data/v1/vad_output_dir

mkdir -p $DATA_DIR
mkdir -p $MFCC_DIR
mkdir -p $VAD_OUTPUT_DIR

ls -1 $AUDIO_CLIPS_DIR/*.wav | awk -F "/" '{p=index($NF,"."); print substr($NF,0,p-1) " " substr($NF,0,p-1);}' > $DATA_DIR/utt2spk
ls -1 $AUDIO_CLIPS_DIR/*.wav | awk -F "/" '{p=index($NF,"."); print substr($NF,0,p-1) " sox -t wav " $0 " -t wav -r 8000 - |";}' > $DATA_DIR/wav.scp

max_jobs_run=20                                                                                                                                                   
sad_num_jobs=4                                                                                                                                                    
sad_opts="--extra-left-context 79 --extra-right-context 21 --frames-per-chunk 150 --extra-left-context-initial 0 --extra-right-context-final 0 --acwt 0.3"         
sad_graph_opts="--min-silence-duration=0.03 --min-speech-duration=0.3 --max-speech-duration=10.0"                                                                  
sad_priors_opts="--sil-scale=0.1"                                                                                                                                  

RUNDIR=`pwd`
cd $KALDI_ROOT/egs/aspire/s5/

# Download Kaldi Aspire VAD model to the current directory
if [ ! -d exp/segmentation_1a ]
then
 wget https://kaldi-asr.org/models/4/0004_tdnn_stats_asr_sad_1a.tar.gz
 tar -xzvf ./0004_tdnn_stats_asr_sad_1a.tar.gz -C $KALDI_ROOT/egs/aspire/s5
fi

utils/utt2spk_to_spk2utt.pl $DATA_DIR/utt2spk > $DATA_DIR/spk2utt
SAD_NET_DIR=exp/segmentation_1a/tdnn_stats_asr_sad_1a

steps/segmentation/detect_speech_activity.sh \
--nj $sad_num_jobs \
            --graph-opts "$sad_graph_opts" \
            --transform-probs-opts "$sad_priors_opts" $sad_opts \
            $DATA_DIR $SAD_NET_DIR \
            $MFCC_DIR $SAD_NET_DIR $VAD_OUTPUT_DIR

cd $RUNDIR