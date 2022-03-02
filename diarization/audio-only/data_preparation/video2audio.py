#! /usr/bin/env python3

import os
import glob
import argparse


def process_video_files(video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(video_dir)
    for filename in glob.glob("*.mp4"):
        basename = filename.split(".")[0]
        os.system(
            f"ffmpeg -i {video_dir}/{filename} -acodec pcm_s16le -ac 1 -ar 16000 {output_dir}/{basename}.wav"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert downloaded video clips to audio clips"
    )
    parser.add_argument(
        "-v", "--video_dir", required=True, help="Directory with video files"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output dir for audio clips"
    )

    args = parser.parse_args()

    process_video_files(args.video_dir, args.output_dir)
