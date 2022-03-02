#! /bin/bash


if [ $# -lt 2 ]; then
    echo "Missing output-dir and stage args"
    exit 1
fi

# -------------------
# Modify according to your paths
PATH=/home/$USER/software/SCTK/bin:$PATH # sclite
PATH=/home/$USER/software/ffmpeg:$PATH
# The following may not be necessary depending on your ffmpeg installation
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/software/libsndfile/lib
GLM=english.glm  # Kaldi GitHub
audioDir=/home/$USER/ego4d_data/wavs_16000
videoDir=/home/$USER/ego4d_data/clips
ego4d=/home/$USER/ego4d_data
vadType="oracle"
# -------------------

outdir=$1
stage=$2
mkdir -p $outdir

if [[ $stage -le 0 ]]; then
    if [ -d $audioDir ]; then
        echo "Audio dir already exists. Exiting"
        echo "If you want to re-run audio extraction, rm $audioDir"
        exit 1;
    fi
    echo "Running audio extraction from videos..."
    bash extract_wav.sh $videoDir $audioDir
fi


if [[ $stage -le 1 ]]; then
    for subset in  "val"; do
        echo "Preparing reference for $subset subset..."
        # Reference TRN
        splitFile=$ego4d/av_${subset}.json
        if [ ! -f $splitFile ]; then
            echo "Split file $splitFile does not exist"
            exit 1;
        fi
        python extract_transcripts_oracle.py $splitFile $outdir/ref.$subset.csv $outdir/ref.$subset.trn
        # GLM norm the trn
        csrfilt.sh -s -i trn $GLM \
                   < $outdir/ref.$subset.trn \
                   > $outdir/ref.$subset.filt.trn
    done
fi

if [[ $stage -le 2 ]]; then
    for subset in "val" ; do
        splitFile=$outdir/ref.$subset.csv
        if [[ $vadType == "oracle" ]]; then
            echo "Using decode_audio_oracle.py"
            python -u decode_audio_oracle.py $splitFile $audioDir \
                   $outdir/hyp.$subset.${vadType}.trn
        else
            echo "Segmentation type $vadType is not supported"
            exit 1
        fi

        # Normalize transcript
        csrfilt.sh -s -i trn $GLM \
                   < $outdir/hyp.$subset.${vadType}.trn \
                   > $outdir/hyp.$subset.${vadType}.filt.trn

        # Score the decoding output
        sclite -r $outdir/ref.$subset.filt.trn trn \
               -h $outdir/hyp.$subset.${vadType}.filt.trn trn \
               -i rm -o pra -o dtl -o sum \
               -n score_${subset}_vad_${vadType}

    done
fi
