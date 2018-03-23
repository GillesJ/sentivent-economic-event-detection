#!/usr/bin/env python3
'''
datahandler.py
sentifmdetect17 
11/24/17
Copyright (c) Gilles Jacobs. All rights reserved.

Module for loading and writing of experiment data.
'''
from glob import glob
import os
from sentifmdetect import settings
from sklearn.model_selection import train_test_split
from sentifmdetect import util
import numpy as np
import itertools
import collections
from datetime import datetime
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import pandas
import json
import logging
from fuzzywuzzy import process, fuzz

util.setup_logging()
pandas.set_option('display.max_colwidth', -1)

def inspect_similar(instances, threshold=90):
    # dupes = {}
    choices = instances[:] # fastest way to copy a list, needed for remove
    for instance in instances:
        choices.remove(instance)
        similar = process.extract(instance, choices, scorer=fuzz.token_sort_ratio, limit=5)
        if similar and similar[0][1] >= threshold: # if found and the most similar is more than threshold
            print(instance)
            print("-----------")
            for sim, score in similar:
                if score >= threshold:
                    print(score, sim)
            print("==================")
                    # dupes[instance].append(sim)

    # for k, v in dupes.items():
    #     print(k)
    #     print("--------")
    #     print("\n".join(v))
    #     print("========")


def clean_data(instances, labels):
    # dupe_idx = util.get_consecutive_duplicate_indices(instances)
    dupe_idx = util.get_duplicate_indices(instances)
    logging.info("\n".join([str((count, item)) for item, count in collections.Counter(instances).items() if count > 1]))
    instances_clean, labels_clean = util.remove_duplicates_by_index(instances, labels, index=dupe_idx)
    instances_clean
    # logging.info(f"Removed {len(dupe_idx)} duplicate instances.")
    return instances_clean, labels_clean

def get_percentage(countdict):
    pctdict = {}
    total = sum(countdict.values())
    for k, v in countdict.items():
        pct = 100*v/total
        pctdict[k] = round(pct, 2)
    return pctdict

def get_label_stats(labels):
    label_stats = {}
    label_stats["multilabel_count"] = dict(collections.Counter(str(l) for l in labels))
    label_stats["multilabel_distr"] = get_percentage(label_stats["multilabel_count"])

    flat_labels = util.flatten(labels)
    label_stats["label_count"] = dict(collections.Counter(flat_labels))
    label_stats["label_distr"] = get_percentage(label_stats["label_count"])
    return label_stats

def unify_dataset(datasets):
    corpus = {}
    # give each instance the appropriate label
    for set_name, inst_lab in datasets.items():
        for i, lab in enumerate(inst_lab[1]):
            if i not in corpus:
                corpus[i] = []
            if lab == '1':
                actual_label = set_name.split("_")[0] # get a nice labelname
                corpus[i].append(actual_label)

    instances = [inst for inst in next(iter(datasets.values()))[0]] # instances should be the same so any random is good
    labels = [corpus[i] if corpus[i] != [] else -1 for i, inst in enumerate(instances)]

    return instances, labels

def load_corpus(dirp):
    try:
        if not os.path.exists(dirp):
            raise ValueError
    except ValueError as e:
        logging.warning(e, "{} is not a dir in your system.".format(dirp))
        sys.exit(1)
    datasets = {}
    for fp in glob(f"{dirp}/*.txt"):
        if fp not in settings.EXCLUDE:
            corpus, labels = load_data(fp)
            dataset_name = os.path.splitext(os.path.basename(fp))[0]
            datasets[dataset_name] = [corpus, labels]

    instances, labels = unify_dataset(datasets)

    instances, labels = clean_data(instances, labels)

    logging.info(f"Loaded {len(instances)} instances, {len(get_label_stats(labels)['label_count'])} labels."
                 f"\nlabel\tcount\tpct\n-------------------\n{get_label_info(labels)}")

    return instances, labels

def load_data(fp):
    corpus = []
    labels = []
    with open(fp, "rt") as f_in:
        for line in f_in:
            line = line.strip()
            if line:
                instance = line.split("\t")[0].strip()
                label = line.split("\t")[1].strip()
                labels.append(label)
                corpus.append(instance)
    assert len(corpus) == len(labels)
    return corpus, labels

def make_train_test_split():
    experiment_data = {
        "meta_split": {
            "function": "sklearn.model_selection.train_test_split",
            "random_state": settings.RANDOM_SEED,
            "test_size": settings.HOLDOUT_SPLIT,
            "shuffle": True,
            "date": settings.TIMESTAMP,
        },
                }
    instances, labels = load_corpus(settings.DATA_DIR)
    x_in, x_out, y_in, y_out, idc_in, idc_out = train_test_split(instances,
                                                                             labels,
                                                                             np.arange(len(labels)),
                                                                             shuffle=True,
                                                                             test_size=settings.HOLDOUT_SPLIT,
                                                                             random_state=settings.RANDOM_SEED)

    logging.info("Train class category counts: \n{}\n---------\n"
          "Test class category counts: \n{}.".format(get_label_info(y_in),
                                                     get_label_info(y_out)))

    experiment_data["holdin_instances"] = x_in
    experiment_data["holdin_labels"] = y_in
    experiment_data["meta_holdin_indices"] = idc_in.tolist()
    experiment_data["holdout_instances"] = x_out
    experiment_data["holdout_labels"] = y_out
    experiment_data["meta_holdout_indices"] = idc_out.tolist()
    # get stats
    experiment_data["meta_stats_alldata"] = get_label_stats(labels)
    experiment_data["meta_stats_holdin"] = get_label_stats(y_in)
    experiment_data["meta_stats_holdout"] = get_label_stats(y_out)

    with open("experiment_data.json", "wt") as exp_out:
        json.dump(experiment_data, exp_out)

    with open("experiment_data.json", "rt") as exp_in_test:
        data_test = json.load(exp_in_test)
    pass


def get_label_info(labels):
    label_stats = get_label_stats(labels)
    stat_info = list(util.common_entries(label_stats["label_count"], label_stats["label_distr"]))
    stat_info.sort(key = lambda x: str(x[0])) # sort labels alphabetically = consistent for comparison
    stat_info_msg = "\n".join([f"{lab}\t{n}\t{pct}" for (lab, n, pct) in stat_info])
    return stat_info_msg

def parse_fold_logs(dirp):
    all_foldlog = []
    fps = glob("{}/*/*json".format(dirp))
    for fp in fps:
        try:
            with open(fp, "rt") as f_in:
                for line in f_in: # line json
                    foldlog = json.loads(line)
                    all_foldlog.append(foldlog)
        except Exception as e:
            logging.exception("Failed to load foldlog {}.".format(fp), e)

    return all_foldlog

def foldlogs_to_dataframe(fl,
                          fold_keys=["f1", "precision", "recall", "accuracy", "auc"],
                          groupby_key=lambda x: (x["clf_type"], x["clf_params"])):
    # unify the list of folds in the {'column': 'index'}
    # group the folds with the same classifier-parameters
    from sentifmdetect import scorer

    def move_results_up(d):
        new_d = {}
        for k, v in d.items():
            if k == "results" and isinstance(v, dict):
                for kk, vv in v.items():
                    new_d[kk] = vv
            else:
                new_d[k] = v
        return new_d

    data = {}
    for k in fold_keys:
        data[k] = []
        data[f"{k}_pred"] = []
        data[f"{k}_per_label"] = []
    data["system"] = []

    groups = itertools.groupby(fl, key=groupby_key)
    for name, fold_group in groups:
        fold_group = list(fold_group)
        data["system"].append(name)
        if len(fold_group) == 6:
            del fold_group[1::2]
        # move any embedded dict up to top level
        fold_group = [move_results_up(foldlog) for foldlog in fold_group]

        # average any scores over the folds
        avg = {k: [] for k in fold_keys}
        frompred = {k: [] for k in fold_keys}
        frompred["auc"] = []
        for f in fold_group:
            predres = scorer.get_metrics(np.array(f["y_true"], dtype=int), np.array(f["y_pred"], dtype=int), average=None)
            frompred["precision"].append(predres["precision"])
            frompred["recall"].append(predres["recall"])
            frompred["f1"].append(predres["f1"])
            frompred["accuracy"].append(predres["accuracy"])
            frompred["auc"].append(predres["auc"])

            for k in fold_keys:
                avg[k].append(f[k])

        for k in fold_keys:
            try:
                if not None in avg[k]:
                    kavg = np.mean(avg[k])
                    predavg = np.mean(frompred[k])
                    test = np.array(frompred[k])
                    perlab = np.mean(test, axis=0)
                else:
                    raise TypeError
            except TypeError as te:
                logging.exception("Failed to take mean.", te)
                kavg = None
                predavg = None
                perlab = None

            data[k].append(kavg)
            data[f"{k}_pred"].append(predavg)
            data[f"{k}_per_label"].append(perlab)

        data["nfoldstrained"] = len(fold_group)

    data.pop("accuracy_pred")
    data.pop("auc_pred")
    data.pop("accuracy_per_label")
    data.pop("auc_per_label")

    df = pandas.DataFrame().from_dict(data)

    return df

def parse_fold_logs_to_dataframe(dirp):
    all_folds = parse_fold_logs(dirp)
    if all_folds:
        df = foldlogs_to_dataframe(all_folds)
        return df
    else:
        return None


if __name__ == "__main__":

    dirp = [
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-27_18-25-41",
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-26_22-40-00",
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-17_18-01-16",
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_22-50-14",
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-16_17-41-54",
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-11-29_20-49-27",
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-20_17-04-23",
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-22_16-59-07", # no pretrained
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-11-30_09-58-15",
        "/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-15_17-28-53",
            ]
    # dirp = glob("/home/gilles/repos/sentifmdetect17/sentifmdetect/output/*")
    for drp in dirp:
        if "_WORDVECTORS" not in drp:
            print(drp)
            df = parse_fold_logs_to_dataframe(drp)
            if isinstance(df, pandas.DataFrame):
                print(df.sort_values(by=['f1'], ascending=False))

    # instances, labels = load_corpus(settings.DATA_DIR)
    # inspect_similar(instances)
    # test_dupes = ["I like cookies!", "I like cookies", "He did not walk into the barber shop to get a haircut however.",
    #               "He took is dog for a walk past the barbershop."]
    # inspect_similar(test_dupes)