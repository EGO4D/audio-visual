import os
import glob
import argparse
import json
import pandas as pd


def read_persons_from_rttm(file_rttm: str):

    rttm_columns = [
        "NA1",
        "uri",
        "NA2",
        "start",
        "duration",
        "NA3",
        "NA4",
        "speaker",
        "NA5",
        "NA6",
    ]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    data = pd.read_csv(
        file_rttm,
        names=rttm_columns,
        dtype=dtype,
        delim_whitespace=True,
        keep_default_na=False,
    )

    persons = []
    for speaker, segments in data.groupby("speaker"):
        voice_regions = []
        for i, row in segments.iterrows():
            voice_regions.append(
                {
                    "start_time": row.start,
                    "end_time": row.start + row.duration,
                    "person": row.speaker,
                }
            )
        persons.append({"person_id": speaker, "voice_segments": voice_regions})

    return persons


def main(rttm_dir: str, output_json_path: str):

    print(f"Converting RTTM files from {os.path.expanduser(rttm_dir)}")
    files_rttm = sorted(glob.glob(os.path.expanduser(rttm_dir) + "/*.rttm"))
    videos = []
    assert len(files_rttm) > 0
    print(f"Converting {len(files_rttm)} files into a JSON")
    for file_path in files_rttm:
        clip_id = os.path.basename(file_path).split(".")[0]
        persons = read_persons_from_rttm(file_path)
        videos.append(
            {
                "video_uid": clip_id,
                "clips": [{"clip_uid": clip_id, "persons": persons}],
            }
        )

    with open(output_json_path, "w") as fw:
        json.dump({"videos": videos}, fw)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert all RTTM files from the input folder to a single JSON for Ego4D audio-only diarization challenge scoring",
        add_help=True,
        usage="%(prog)s [options]",
    )

    parser.add_argument(
        "-r",
        "--rttm_dir",
        nargs=None,
        metavar="STR",
        dest="rttm_dir",
        help="Input directory with RTTM files",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs=None,
        metavar="STR",
        dest="output_json_path",
        help="Path to the output JSON",
    )

    args = parser.parse_args()
    main(args.rttm_dir, args.output_json_path)
