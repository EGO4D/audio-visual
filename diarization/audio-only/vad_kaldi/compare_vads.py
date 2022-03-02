import argparse
import glob
import os


def list_sum_from_vad_file(filename: str):

    out = []
    sum = 0.0
    for line in open(filename):
        start, end, _ = line.split(maxsplit=2)
        out.append((float(start), float(end)))
        sum += float(end) - float(start)
    return out, sum


def compare_vad_files(filename1: str, filename2: str):

    arr1, sum1 = list_sum_from_vad_file(filename1)
    arr2, sum2 = list_sum_from_vad_file(filename2)

    i = j = 0
    intersection = 0.0

    while i < len(arr1) and j < len(arr2):
        l = max(arr1[i][0], arr2[j][0])
        r = min(arr1[i][1], arr2[j][1])

        if l <= r:
            intersection += r - l

        if arr1[i][1] < arr2[j][1]:
            i += 1
        else:
            j += 1

    return sum1, sum2, intersection


def main(dir1, dir2):

    print(f"Comparing {os.path.expanduser(dir1)} and {os.path.expanduser(dir2)}")

    files1 = sorted(glob.glob(os.path.expanduser(dir1) + "/*.lab"))
    files2 = sorted(glob.glob(os.path.expanduser(dir2) + "/*.lab"))

    assert len(files1) == len(files2)

    print(f"Analyzing {len(files1)} pairs of files")

    sum_sum1 = sum_sum2 = sum_intersection = 0.0
    for i in range(len(files1)):
        sum1, sum2, intersection = compare_vad_files(files1[i], files2[i])
        sum_sum1 += sum1
        sum_sum2 += sum2
        sum_intersection += intersection

    print(f"Dir1 VAD sum: {sum_sum1:.2f} sec  Dir2 VAD sum: {sum_sum2:.2f} sec")
    print(f"VAD intersection abs: {sum_intersection:.2f} sec")
    print(
        f"VAD intersection rel Dir1: {100*sum_intersection/sum_sum1:.2f} %   VAD intersection rel Dir2: {100*sum_intersection/sum_sum2:.2f} %"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute overlap in corresponding .lab VAD files from two directories",
        add_help=True,
        usage="%(prog)s [options]",
    )

    parser.add_argument(
        "--dir1",
        nargs=None,
        metavar="STR",
        dest="dir1",
        help="First directory",
    )
    parser.add_argument(
        "--dir2",
        nargs=None,
        metavar="STR",
        dest="dir2",
        help="Second directory",
    )

    args = parser.parse_args()
    main(args.dir1, args.dir2)
