#!/usr/bin/env python3

import argparse
import os
import json
import pandas as pd
from typing import List


def sort_rttm(lines: List) -> List:
    return sorted(lines, key=lambda line: float(line.split()[3]))


def get_person_info_from_json(json_path: str) -> pd.DataFrame:
    av_json = json.load(open(json_path))
    csv_rows = []
    for video in av_json["videos"]:
        video_uid = video["video_uid"]
        for clip in video["clips"]:
            for person in clip["persons"]:
                row = {"video_uid": video_uid, "person_id": person["person_id"]}
                for clip_field in [
                    "clip_uid",
                    "parent_start_sec",
                    "parent_end_sec",
                    "parent_start_frame",
                    "parent_end_frame",
                ]:
                    try:
                        row[clip_field] = clip[clip_field]
                    except:
                        pass
                row["voice_segments"] = person["voice_segments"]
                csv_rows.append(row)
    return pd.DataFrame.from_dict(csv_rows)


def df2rttms(df, args):

    for file_id in df["clip_uid"].unique():
        rttm_lines = []
        voice_segments = []
        for _, row in df.loc[df["clip_uid"] == file_id].iterrows():
            if len(row["voice_segments"]) > 0:
                for elem in row["voice_segments"]:
                    if "start_time" in elem and "end_time" in elem and "person" in elem:
                        start_time = float(elem["start_time"])
                        end_time = float(elem["end_time"])
                        label = elem["person"]
                        if label != ":-1":  # Skip :-1 which refers to sources of noise
                            if start_time != end_time:
                                rttm_lines.append(
                                    "SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>".format(
                                        file_id,
                                        start_time,
                                        end_time - start_time,
                                        label,
                                    )
                                )
                                voice_segments.append(
                                    (float(start_time), float(end_time))
                                )
                with open(
                    os.path.join(args.rttm_output_dir, file_id + ".rttm"), "w"
                ) as rttm_out_file:
                    for line in sort_rttm(rttm_lines):
                        print(line, file=rttm_out_file)
                last_segment_start = None
                last_segment_end = None
                with open(
                    os.path.join(args.lab_output_dir, file_id + ".lab"), "w"
                ) as lab_out_file:
                    for start, end in sorted(voice_segments):
                        if last_segment_end is None:
                            last_segment_start = start
                            last_segment_end = end
                            continue
                        elif last_segment_end >= start:
                            last_segment_end = max(last_segment_end, end)
                            last_segment_start = min(last_segment_start, start)
                            continue
                        else:
                            print(
                                f"{last_segment_start:.3f} {last_segment_end:.3f} speech",
                                file=lab_out_file,
                            )
                            last_segment_start = start
                            last_segment_end = end
                    if last_segment_end is not None:
                        print(
                            f"{last_segment_start:.3f} {last_segment_end:.3f} speech",
                            file=lab_out_file,
                        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate RTTM files from JSON.",
        add_help=True,
        usage="%(prog)s [options]",
    )

    parser.add_argument(
        "-j",
        "--json-filename",
        nargs=None,
        metavar="STR",
        dest="json_filename",
        help="Filename of csv file.",
    )
    parser.add_argument(
        "-o",
        "--rttm-output-dir",
        nargs=None,
        metavar="STR",
        dest="rttm_output_dir",
        help="Directory where RTTMs will be placed.",
    )
    parser.add_argument(
        "-l",
        "--lab-output-dir",
        nargs=None,
        metavar="STR",
        dest="lab_output_dir",
        help="Directory where oracle VAD files will be placed.",
    )

    args = parser.parse_args()
    os.makedirs(args.rttm_output_dir, exist_ok=True)
    os.makedirs(args.lab_output_dir, exist_ok=True)

    df_json = get_person_info_from_json(args.json_filename)
    df2rttms(df_json, args)
