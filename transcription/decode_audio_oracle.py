import os
import numpy as np
import soundfile
from espnet2.bin.asr_inference import Speech2Text
from asr_utils import get_clip2utt


def transcribe_oracle_segment(speech2text, speech, rate, tb, te, min_length=7):
    Nbeg = int(float(tb) * rate)
    Nend = int(float(te) * rate)
    if Nend - Nbeg < min_length:
        print("Utterance is too short to evaluate.")
        return ''
    try:
        nbests = speech2text(speech[Nbeg:Nend])
        text, *_ = nbests[0]
        text = text.strip()
        text = text.lower().replace("<sos/eos>", "")
    except:
        text = ''
        print("Error in processing segment")

    return text

def transcribe_segmented(speech2text, speech, rate, segment_length=5.0):
    text_list = []
    N = int(np.size(speech))
    Nhop = int(segment_length * rate)

    for sample_begin in np.arange(0, N-Nhop+1, Nhop):
        sample_end = sample_begin + Nhop
        nbests = speech2text(speech[sample_begin:sample_end])
        text, *_ = nbests[0]
        text = text.strip()
        if text != "":
            text_list.append(text)

    result_text = " ".join(text_list)
    return result_text.strip().lower().replace("<sos/eos>", "")


def main(args):
    assert(os.path.exists(args.audio_dir))
    speech2text = Speech2Text.from_pretrained(
        "Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave",
        # Decoding parameters are not included in the model file
        maxlenratio=0.0,
        minlenratio=0.0,
        beam_size=20,
        ctc_weight=0.3,
        lm_weight=0.5,
        penalty=0.0,
        nbest=1
    )

    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text

    clip2utt = get_clip2utt(args.split_file)
    print(len(clip2utt))
    with open(args.hyp_trn, 'w') as trn_f:
        ctr, empty_ctr = 0, 0
        for clip in clip2utt:
            print(clip)
            audio_file = os.path.join(args.audio_dir, clip+'.wav')
            speech, rate = soundfile.read(audio_file)
            assert(rate == 16000)
            for utt in clip2utt[clip]:
                hyp_text = transcribe_oracle_segment(
                    speech2text, speech, rate,
                    utt.tb, utt.te,
                    min_length=args.min_length
                )

                trn_f.write("{} ({})\n".format(hyp_text, utt.name))
                ctr += 1
                if hyp_text.strip() == '':
                    empty_ctr += 1

        print("Processed {} clips, {} of which are empty".format(ctr, empty_ctr))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "split_file",
        type=str,
        help="CSV file that contains train/val/test set info"
    )
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Directory of the extracted 16kHz audio files"
    )
    parser.add_argument(
        "hyp_trn",
        type=str,
        help="Output TRN file that will contain the ASR hypotheses"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        help="Minimum number of frames to start decoding, "
        "otherwise the segment is too short to evaluate",
        default=7
    )
    args = parser.parse_args()
    main(args)
