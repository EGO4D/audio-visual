import os, sys, subprocess
import tempfile
import json
from datetime import datetime
from . import sctk_bin_path


MAX_WER = 1000000
DEFAULT_FIELDS = ["channel", "tb", "dur", "word"]
IGNORE_SEGMENT_STR = "IGNORE_TIME_SEGMENT_IN_SCORING"
GLM = "english.glm"


def get_wer_from_score_file(score_file):
    with open(score_file, 'r') as f:
        for line in f:
            if line.find("Sum/Avg") != -1:
                wer = line.strip().split()[9]
                print(line, wer)
                return wer

    return MAX_WER


def cleanup_score_dir(submission_name):
    f_list = [f.name for f in os.scandir(submission_name)]
    for f in f_list:
        print(f)
        os.remove(os.path.join(submission_name, f))

    print("Removing ", submission_name)
    os.rmdir(submission_name)


def compute_wer(ref_stm, hyp_ctm, submission_name):
    wer = MAX_WER
    assert(submission_name == os.path.dirname(hyp_ctm)), "Hypothesis file is not located correctly"

    cur_time = datetime.now()
    time_str = cur_time.strftime("%m-%d-%Y-%H-%M-%S")
    print("Current time:", time_str)
    filename = submission_name+"/score_{}.sys".format(time_str)
    try:
        os.remove(filename)
        print("Removing", filename)
    except OSError:
        pass

    sys.path.insert(0, (sctk_bin_path))
    subprocess.check_call(
        [
            "{}/sclite".format(sctk_bin_path),
            "-r", ref_stm, "stm",
            "-h", hyp_ctm, "ctm",
            "-m", "ref",
            "-o", "sum",
            "-n", "score_{}".format(time_str)
        ],
        env={**os.environ, 'PATH': ':'.join(sys.path)}
    )

    if os.path.exists(filename):
        wer = get_wer_from_score_file(filename)
    else:
        print("Problem in producing the scoring file.")
        exit(1)
    return wer


def filter_trn(in_trn, filt_trn, trn_format, glm=GLM):
    sys.path.insert(0, sctk_bin_path)

    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    subprocess.run(
        "which egrep",
        env={**os.environ},
        shell=True, stdout=temp_file
    )
    temp_file.close()

    egrep_path = None
    with open(temp_file.name ,'r')  as f:
        egrep_path = os.path.dirname(f.read().strip())

    os.remove(temp_file.name)
    assert(egrep_path is not None), "System does not have egrep"
    sys.path.append(egrep_path)

    filter_cmd = "{}/csrfilt.sh -s -i {} {} < {} > {}".format(
        sctk_bin_path,
        trn_format,
        glm,
        in_trn,
        filt_trn
    )
    subprocess.run(
        filter_cmd,
        env={**os.environ, 'PATH': ':'.join(sys.path)},
        shell=True
    )
    subprocess.run(
        "echo $PATH > /code/path.txt",
        env={**os.environ, 'PATH': ':'.join(sys.path)},
        shell=True
    )

    return os.path.exists(filt_trn)


def json_to_stm(json_in, submission_name):
    av_json = {}
    with open(json_in, 'r') as f:
        av_json = json.load(f)
    assert(len(av_json)), "Reference file is corrupted"

    clip2trn = {}
    with open(submission_name+"/ref.stm", "w") as stm:
        for clip in sorted(av_json):
            trn_list = av_json[clip]
            assert(len(trn_list))
            sorted_trn_list = sorted(trn_list, key=lambda x: float(x["tb"]))
            for trn in sorted_trn_list:
                stm.write("{} {} {} {} {} {}\n".format(
                    clip,
                    trn["channel"],
                    trn["speaker"],
                    trn["tb"],
                    trn["te"],
                    trn["txt"].lower()
                )
                          )

    # Apply GLM filtering
    is_filtered = filter_trn(submission_name+"/ref.stm", submission_name+"/ref.filt.stm", "stm")
    assert(is_filtered), "Could not apply GLM filtering to the ref"
    return submission_name+'/ref.filt.stm'


def json_to_ctm(json_in, submission_name):
    results_dict = {}
    with open(json_in, 'r') as f:
        results_dict = json.load(f)

    assert(len(results_dict)), "Empty set of results"
    # check for validity of input dict schema
    with open(submission_name+'/hyp.ctm', 'w') as ctm:
        for clip_uid in sorted(results_dict):
            word_entries = results_dict[clip_uid]
            assert(type(word_entries) is list), "Entries should be in a list"
            for entry in word_entries:
                for field in DEFAULT_FIELDS:
                    assert(
                        field in entry
                    ), "{} is missing from the entry {}".format(
                        field, entry
                    )
                ctm.write("{} {} {} {} {}\n".format(
                    clip_uid, entry["channel"], entry["tb"],
                    entry["dur"], entry["word"].replace("\n", "").lower()
                ))

    # Apply GLM filtering
    is_filtered = filter_trn(submission_name+"/hyp.ctm", submission_name+"/hyp.filt.ctm", "ctm")
    assert(is_filtered), "Could not apply GLM filtering to the hyp"
    return submission_name+'/hyp.filt.ctm'


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    sys.path.insert(0, ("/usr/bin/"))
    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "train_split": {
                    "WER": MAX_WER,
                    "Total": MAX_WER,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        # print("Check input format")

        try:
            submission_metadata = kwargs['submission_metadata']
            submission_name = "submission_by_team{}".format(
                submission_metadata["participant_team"]
            )
        except:
            submission_name = "submission_by_test0"

        submission_name = submission_name.replace('/','')
        if not os.path.exists(submission_name):
            os.mkdir(submission_name)

        print("Ref: JSON 2 STM")
        ref_stm = json_to_stm(test_annotation_file, submission_name)
        print("Hyp: JSON 2 CTM")
        hyp_ctm = json_to_ctm(user_submission_file, submission_name)

        print("Compute WER...")
        wer = compute_wer(ref_stm, hyp_ctm, submission_name)
        cleanup_score_dir(submission_name)

        output["result"] = [
            {
                "test_split": {
                    "WER": wer,
                    "Total": wer,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]['test_split']
        print("Completed evaluation for Test Phase")
    return output
