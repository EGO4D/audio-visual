#!/bin/bash
set -e

export KALDI_ROOT=~/repos/kaldi

VBX_DIR=~/repos/VBx # Path to the root dir of VBx installation
METHOD=AHC+VB
SKIP_XVECTORS=0 # Set to 1 if you want to skip xvector extraction (i.e. extracted already) 

EXP_DIR=~/ego4d_data/v1/vbx_diarization # output experiment directory
XVEC_DIR=$EXP_DIR/xvectors # output xvectors directory
WAV_DIR=~/ego4d_data/v1/audio_clips # wav files directory
FILE_LIST=$EXP_DIR/list.txt # txt list of files to process
LAB_DIR=~/ego4d_data/v1/vad_output_dir # lab files directory with VAD segments generated by KALDI
RTTM_DIR=~/ego4d_data/v1/rttms # reference rttm files directory

mkdir -p $EXP_DIR
ls -1 $WAV_DIR/*.wav | awk -F "/" '{p=index($NF,"."); print substr($NF,0,p-1)}' > $EXP_DIR/list.txt

# Xvector extraction
if [ "$SKIP_XVECTORS" -eq "0" ]; then
	WEIGHTS_DIR=$VBX_DIR/VBx/models/ResNet101_16kHz/nnet
	if [ ! -f $WEIGHTS_DIR/raw_81.pth ]; then
	    cat $WEIGHTS_DIR/raw_81.pth.zip.part* > $WEIGHTS_DIR/unsplit_raw_81.pth.zip
		unzip $WEIGHTS_DIR/unsplit_raw_81.pth.zip -d $WEIGHTS_DIR/
fi

	WEIGHTS=$VBX_DIR/VBx/models/ResNet101_16kHz/nnet/raw_81.pth
	EXTRACT_SCRIPT=$VBX_DIR/VBx/extract.sh
	DEVICE=cpu

	mkdir -p $XVEC_DIR
	$EXTRACT_SCRIPT ResNet101 $WEIGHTS $WAV_DIR $LAB_DIR $FILE_LIST $XVEC_DIR $DEVICE

	bash $XVEC_DIR/xv_task
 fi

# Diarization
BACKEND_DIR=$VBX_DIR/VBx/models/ResNet101_16kHz
	TASKFILE=$EXP_DIR/diar_"$METHOD"_task
	OUTFILE=$EXP_DIR/diar_"$METHOD"_out
	rm -f $TASKFILE $OUTFILE
	mkdir -p $EXP_DIR/lists

	thr=-0.015
	smooth=7.0
	lda_dim=128
	Fa=0.4
	Fb=8
	loopP=0.0
	OUT_DIR=$EXP_DIR/out_dir_"$METHOD"
		mkdir -p $OUT_DIR
		while IFS= read -r line; do
			grep $line $FILE_LIST > $EXP_DIR/lists/$line".txt"
			echo "python3 $VBX_DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file $XVEC_DIR/xvectors/$line.ark --segments-file $XVEC_DIR/segments/$line --plda-file $BACKEND_DIR/plda --xvec-transform $BACKEND_DIR/transform.h5 --threshold $thr --init-smoothing $smooth --lda-dim $lda_dim --Fa $Fa --Fb $Fb --loopP $loopP" >> $TASKFILE
		done < $FILE_LIST
		bash $TASKFILE > $OUTFILE

		# Scoring
		cat $OUT_DIR/rttms/*.rttm > $OUT_DIR/sys.rttm
		cat $RTTM_DIR/*.rttm > $OUT_DIR/ref.rttm
		$VBX_DIR/dscore/score.py --collar 0.25 --ignore_overlaps -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_forgiving
		$VBX_DIR/dscore/score.py --collar 0.25 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_fair
		$VBX_DIR/dscore/score.py --collar 0.0 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_full
    