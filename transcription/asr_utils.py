import sys
from collections import defaultdict

class Utterance:
    def __init__(self, name, spk, tb, te):
        self.name = name
        self.tb = tb
        self.te = te
        self.spk = spk


def get_list_of_clips(split_file):
    clip_list = []
    with open(split_file, 'r')  as f:
        for line in f:
            try:
                clip = line.strip().split(',', 1)[0]
                clip_list.append(clip)
            except:
                print("Input format must be a CSV")
                exit(1)

    return clip_list

def get_clip2utt(split_file):
    clip2utt = defaultdict(list)
    with open(split_file, 'r') as f:
        for line in f:
            try:
                entry = line.strip().split(',')
                clip = entry[1]
                clip2utt[clip].append(
                    Utterance(entry[0], entry[2], entry[3], entry[4])
                )

            except:
                print("Check the input format")
                exit(1)

    return clip2utt
