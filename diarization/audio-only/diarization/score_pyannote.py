import os
import glob
import argparse
import statistics
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm


def main(ref_dir: str, hyp_dir: str):

    print(f"Comparing {os.path.expanduser(ref_dir)} and {os.path.expanduser(hyp_dir)}")

    files_ref = sorted(glob.glob(os.path.expanduser(ref_dir) + "/*.rttm"))
    files_hyp = sorted(glob.glob(os.path.expanduser(hyp_dir) + "/*.rttm"))

    assert len(files_ref) == len(files_hyp)

    print(f"Analyzing {len(files_ref)} pairs of files")

    DERs = []
    metric = DiarizationErrorRate()
    for i in range(len(files_ref)):
        print(files_ref[i])
        segments_ref = list(load_rttm(files_ref[i]).values())[0]
        segments_hyp = list(load_rttm(files_hyp[i]).values())[0]
        der = metric(segments_ref, segments_hyp)  # , uem=Segment(0, 3000))
        print(der)
        DERs.append(der)

    print("Average DER = ", statistics.mean(DERs))
    metric.report(display=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute mean DER over a set of ref and hyp RTTMs",
        add_help=True,
        usage="%(prog)s [options]",
    )

    parser.add_argument(
        "-r",
        "--ref_dir",
        nargs=None,
        metavar="STR",
        dest="ref",
        help="Directory with reference RTTMs",
    )
    parser.add_argument(
        "-y",
        "--hyp_dir",
        nargs=None,
        metavar="STR",
        dest="hyp",
        help="Directory with hypothesized RTTMs",
    )

    args = parser.parse_args()
    main(args.ref, args.hyp)
