#!/usr/bin/env python3
'''
reporter.py
sentifmdetect17 
3/26/18
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
from sentifmdetect import util
import numpy as np
from collections import OrderedDict
import os
np.set_printoptions(precision=2)

def average_best_onevsrest():
    """
    Averages the best results of onevsrest.
    [0.95, 0.91, 0.93], #  BuyRating L-SVM
    [0.62, 0.73, 0.67], #  Debt L-SVM
    [0.50, 1.00, 0.67], #  Dividend L-SVM
    [0.58, 0.44, 0.50], #  MergerAcquisition LSTM
    [0.80, 0.76, 0.78], #  Profit RBF-SVM
    [0.83, 0.56, 0.67], #  QuarterlyResults L-SVM
    [0.88, 0.75, 0.81], #  SalesVolume L-SVM
    [1.00, 0.50, 0.67], #  ShareRepurchase L-SVM
    [1.00, 1.00, 1.00], #  TargetPrice LSTM
    [0.91, 0.77, 0.83], #  Turnover L-SVM
    :return:
    """

    best_onevsrest = np.array(
        [
            [0.95, 0.91, 0.93], #  BuyRating L-SVM
            [0.62, 0.73, 0.67], #  Debt L-SVM
            [0.50, 1.00, 0.67], #  Dividend L-SVM
            [0.58, 0.44, 0.50], #  MergerAcquisition LSTM
            [0.80, 0.76, 0.78], #  Profit RBF-SVM
            [0.83, 0.56, 0.67], #  QuarterlyResults L-SVM
            [0.88, 0.75, 0.81], #  SalesVolume L-SVM
            [1.00, 0.50, 0.67], #  ShareRepurchase L-SVM
            [1.00, 1.00, 1.00], #  TargetPrice LSTM
            [0.91, 0.77, 0.83], #  Turnover L-SVM
        ]
    )

    avg = np.mean(best_onevsrest, axis=0)

    print(f"One vs rest average best P R F1: {avg}")

def print_avg_results():
    '''
    NOTE These LSTM results have rounding error, the correct average results are in the paper!
    '''
    svm_linearkernel = {
    "BuyRating": [0.952380952381, 0.909090909091,],
    "Debt": [0.50, 1.00,],
    "Dividend": [0.615384615385, 0.727272727273,],
    "MergerAcquisition": [0.555555555556, 0.40,],
    "Profit": [0.754385964912, 0.741379310345,],
    "QuarterlyResults": [0.818181818182, 0.529411764706,],
    "SalesVolume": [0.883720930233, 0.745098039216,],
    "ShareRepurchase": [1.00, 0.50,],
    "TargetPrice": [1.00, 0.75,],
    "Turnover": [0.909090909091, 0.769230769231,],
    }

    svm_optimisedrbf = {
    "BuyRating": [0.952380952381, 0.909090909091,],
    "Debt": [0.50, 1.00,],
    "Dividend": [0.538461538462, 0.636363636364,],
    "MergerAcquisition": [0.00, 0.00,],
    "Profit": [0.80, 0.758620689655,],
    "QuarterlyResults": [0.826086956522, 0.558823529412,],
    "SalesVolume": [0.942857142857, 0.647058823529,],
    "ShareRepurchase": [1.00, 0.50,],
    "TargetPrice": [1.00, 0.75,],
    "Turnover": [0.869565217391, 0.769230769231,],
    }
    # ^\s*([a-zA-Z]+)\s+(\d\.\d\d)\s+(\d\.\d\d)\s+\d\.\d\d\s+\d+ | "$1": [$2, $3],
    lstm_6b = {
        "BuyRating": [0.86, 0.82],
        "Debt": [0.00, 0.00],
        "Dividend": [0.50, 0.55],
        "MergerAcquisition": [0.40, 0.32],
        "Profit": [0.82, 0.79],
        "QuarterlyResults": [0.77, 0.68],
        "SalesVolume": [0.84, 0.73],
        "ShareRepurchase": [1.00, 0.67],
        "TargetPrice": [0.75, 0.75],
        "Turnover": [0.90, 0.73],
    }

    lstm_sentifm = {
        "BuyRating": [0.91, 0.91],
        "Debt": [1.00, 0.50],
        "Dividend": [0.50, 0.36],
        "MergerAcquisition": [0.32, 0.24],
        "Profit": [0.75, 0.81],
        "QuarterlyResults": [0.87, 0.38],
        "SalesVolume": [0.92, 0.67],
        "ShareRepurchase": [0.80, 0.67],
        "TargetPrice": [1.00, 0.50],
        "Turnover": [0.95, 0.69],
    }

    lstm_no = {
        "BuyRating": [0.81, 0.59],
        "Debt": [0.33, 0.50],
        "Dividend": [0.75, 0.55],
        "MergerAcquisition": [0.21, 0.12],
        "Profit": [0.83, 0.33],
        "QuarterlyResults": [0.67, 0.35],
        "SalesVolume": [0.86, 0.61],
        "ShareRepurchase": [0.60, 0.50],
        "TargetPrice": [1.00, 0.50],
        "Turnover": [0.88, 0.58],
    }

    lstm_onevsrest = {
        "BuyRating": [0.88, 0.95],
        "Debt": [0.50, 0.50],
        "Dividend": [0.55, 0.55],
        "MergerAcquisition": [0.58, 0.44],
        "Profit": [0.81, 0.74],
        "QuarterlyResults": [0.84, 0.47],
        "SalesVolume": [0.81, 0.76],
        "ShareRepurchase": [0.75, 0.50],
        "TargetPrice": [1.00, 1.00],
        "Turnover": [0.94, 0.65],
    }

    all = OrderedDict(
        [
            ("svm_linearkernel", svm_linearkernel),
            ("svm_optimisedrbf", svm_optimisedrbf),
            ("lstm_holdin", lstm_sentifm),
            ("lstm_6b", lstm_6b),
            ("lstm_nopretrained", lstm_no),
            ("lstm_onevsrest", lstm_onevsrest),
        ]
    )

    for system, data in all.items():
        f1_all = []
        p_all = []
        r_all = []
        for k, v in data.items():
            if v[0] == 0 and v[1] == 0:
                f1 = 0.0
            else:
                f1 = 2*(v[0]*v[1])/(v[0]+v[1])
            f1_all.append(f1)
            p_all.append(v[0])
            r_all.append(v[1])
        print(f"\n{system}")
        print(f"precision avg: {format(np.mean(p_all), '.2f')}")
        print(f"recall avg: {format(np.mean(r_all), '.2f')}")
        print(f"f1 avg: {format(np.mean(f1_all), '.2f')}")

def get_best_all(data):
    '''
    Get the best holdout score parametrization over all runs.
    :param data:
    :return:
    '''
    tasks = {}
    for run_name, run_data in experiments.items():
        if "binary_task" in run_data:
            for task, data in run_data["binary_task"].items():
                if task in tasks:
                    tasks[task].append(data)
                else:
                    tasks[task] = [data]

    best = {}
    for task, data in tasks.items():
        best_data = max(data, key = lambda x: x["scores"]["f1"])
        best[task] = best_data

    return best

if __name__ == "__main__":

    average_best_onevsrest()

    expmetadata = {
        "buyrating_partial": "/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-26_17:51:11_CEST/metadata.json",
        "rest": "/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-29_18:25:07_CEST/metadata.json",
        "fullplus": "/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-31_00:54:36_CEST/metadata.json",
        "4failed": "/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-04-02_13:50:21_CEST/metadata.json",
        "5retry": "/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-04-02_23:00:22_CEST/metadata.json",
    }
    optdir = "/home/gilles/repos/sentifmdetect17/results"
    best_fp = os.path.join(optdir, "best_bin.json")

    if os.path.isfile(best_fp):
        best = util.read_json(best_fp)

    else:
        experiments = {}
        for run_name, fp in expmetadata.items():
            experiments[run_name] = util.read_metadata(fp)

        # test if out is same over runs
        l_out = []
        i_out = []
        for run_name, data in experiments.items():
            labels_out = data["feature_data"]["labels_out"]
            inst_out = data["feature_data"]["instances_out"]
            if labels_out not in l_out:
                l_out.append(labels_out)
            if inst_out not in i_out:
                i_out.append(inst_out)
        assert len(l_out) == 1
        assert len(i_out) == 1

        best = get_best_all(experiments)
        best.update({"inst_out": i_out[0], "labels_out": l_out[0]})
        util.write_json(best, best_fp)

    for task, data in best.items():
        if task not in ["inst_out", "labels_out"]:
            n_pos = dict(zip(*np.unique(data['y_out'], return_counts=True)))[1]
            print(f"{task} & {format(data['scores']['precision'], '.2f')} & {format(data['scores']['recall'], '.2f')} & {format(data['scores']['f1'], '.2f')} \\\\")
            # print(f"{task} (n={n_pos}) & {format(data['scores']['precision'], '.2f')} & {format(data['scores']['recall'], '.2f')} & {format(data['scores']['f1'], '.2f')}\\\\")
            # print(format(data['best_cv_score'], '.2f'))
    f1_avg = np.mean([data["scores"]["f1"] for k, data in best.items() if k not in ["inst_out", "labels_out"]])
    p_avg = np.mean([data["scores"]["precision"] for k, data in best.items() if k not in ["inst_out", "labels_out"]])
    r_avg = np.mean([data["scores"]["recall"] for k, data in best.items() if k not in ["inst_out", "labels_out"]])
    acc_avg = np.mean([data["scores"]["accuracy"] for k, data in best.items() if k not in ["inst_out", "labels_out"]])
    print(f"\\hline\navg & {format(p_avg, '.2f')} & {format(r_avg, '.2f')} & {format(f1_avg, '.2f')}  \\\\ \\hline \\hline")
