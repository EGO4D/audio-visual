#!/usr/bin/env python3

import os
import pandas as pd
import argparse


def kaldi_segments_to_labs(vad_output_dir: str):

    expected_segment_path = os.path.join(vad_output_dir + "_seg", "segments")

    if not os.path.isfile(expected_segment_path):
        raise FileNotFoundError(
            f"Kaldi VAD segment file {expected_segment_path} does not exist"
        )

    if not os.path.isdir(vad_output_dir):
        print(
            f"Warning: Directory {vad_output_dir} should exist, is your setup correct?"
        )
        os.makedirs(vad_output_dir)

    df = pd.read_csv(
        expected_segment_path,
        names=["segment_id", "file_id", "start", "end"],
        delimiter=" ",
    )
    print("Processing ids:")
    for file_id in df["file_id"].unique():
        print(file_id)
        if not file_id:
            continue
        row_buffer = None
        with open(os.path.join(vad_output_dir, file_id + ".lab"), "w") as fw:
            for i, row in df.loc[df["file_id"] == file_id].iterrows():
                if row_buffer is None:
                    row_buffer = row
                    continue
                elif row_buffer["end"] == row["start"]:
                    row_buffer["end"] = row["end"]
                    continue
                else:
                    print(f"{row_buffer['start']} {row_buffer['end']} speech", file=fw)
                    row_buffer = row
            if row_buffer is not None:
                print(f"{row_buffer['start']} {row_buffer['end']} speech", file=fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take audio-visual annotation json and original video files and generates corresponding audio clips"
    )
    parser.add_argument(
        "VAD_output_dir", help="Path to the dir used for VAD output in run_vad.sh"
    )
    args = parser.parse_args()

    kaldi_segments_to_labs(args.VAD_output_dir)
