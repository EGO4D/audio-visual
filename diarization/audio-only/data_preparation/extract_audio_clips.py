#! /usr/bin/env python3

import json
import pandas as pd
import os
import glob
import argparse


def get_clip_info_from_json(json_path: str) -> pd.DataFrame:
    av_json = json.load(open(json_path))

    csv_rows = []
    for video in av_json["videos"]:
        video_uid = video["video_uid"]
        for clip in video["clips"]:
            row = {
                "video_uid": video_uid,
            }

            for clip_field in [
                "clip_uid",
                "parent_start_sec",
                "parent_end_sec",
                "parent_start_frame",
                "parent_end_frame",
            ]:
                row[clip_field] = clip[clip_field]

            csv_rows.append(row)

    return pd.DataFrame.from_dict(csv_rows)


def process_video_files(df, video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(video_dir)
    for filename in glob.glob("*.mp4"):
        basename = filename.split(".")[0]
        for _, row in df.loc[df["video_uid"] == basename].iterrows():
            print(row["clip_uid"], row["parent_start_sec"], row["parent_end_sec"])
            os.system(
                f'ffmpeg -i {video_dir}/{filename} -ss {row["parent_start_sec"]} -to {row["parent_end_sec"]} -acodec pcm_s16le -ac 1 -ar 16000 {output_dir}/{row["clip_uid"]}.wav'
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Take audio-visual annotation json and original video files and generates corresponding audio clips"
    )
    parser.add_argument(
        "-j", "--json_path", required=True, help="Path to the JSON with annotations"
    )
    parser.add_argument(
        "-v", "--video_dir", required=True, help="Directory with video files"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output dir for audio clips"
    )

    args = parser.parse_args()

    df = get_clip_info_from_json(args.json_path)
    process_video_files(df, args.video_dir, args.output_dir)
