#!/usr/bin/env python3
'''
scorer.py
sentifmdetect17 
11/28/17
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
from sentifmdetect import settings
from sentifmdetect import util
from sklearn import metrics
import json
import os
import logging
import multiprocessing
from sklearn.externals.joblib import dump

util.setup_logging()

def log_metrics_and_params(clf, results, fold_log_dir, y_true=[], y_pred=[], model_save_fp=None, log_name=None):
    # log results and save path
    clf_type = type(clf)
    try:
        clf_params = clf.get_params()
    except TypeError as e:
        logging.info(e)
        clf_params = clf.sk_params
    logging.info('\nResults:\t{}\nPipeline:\t{}\n'.format(results, clf_params))
    to_write = {}
    to_write['pipe_name'] = log_name
    to_write['results'] = results
    to_write['y_pred'] = y_pred.tolist()
    to_write['y_true'] = y_true.tolist()
    to_write['clf_type'] = str(clf_type)
    to_write['clf_params'] = str(clf_params)
    to_write['savepath_model'] = model_save_fp
    to_write['n_folds'] = settings.KFOLD
    # to_write['cv_object'] = str(settings.CV) # TODO write all relevant metadata to foldlog, maybe just write the settings to to foldlogdir
    # pprint(to_write)
    try:
        current_proc = multiprocessing.current_process()
        proc_str = "{}{}{}".format(current_proc.name, current_proc._identity, current_proc.pid)
        intermed_result_proc_fp = os.path.join(fold_log_dir, "{}_{}.json".format(settings.TIMESTAMP, proc_str))
        with open(intermed_result_proc_fp, mode='a') as int_f:
            json.dump(to_write, int_f, sort_keys=True)
            int_f.write("{}".format(os.linesep))
    except Exception as e:
        logging.exception("Could not write intermediate result.")

def save_model(clf, fold_model_dir, log_name=None):
    # save model with timestamp
    if log_name:
        savepath_suffix = '{}_{}'.format(settings.TIMESTAMP, log_name)
    else:
        savepath_suffix = '{}'.format(settings.TIMESTAMP)

    model_savepath = os.path.join(fold_model_dir, 'model_{}.pkl'.format(savepath_suffix))
    try:
        dump(clf, model_savepath, compress=1)
    except Exception as e:
        logging.exception("Failed to pickle candidate classifier.")

    return model_savepath

def get_metrics(y_true=[], y_pred=[], average=settings.SCORE_AVERAGING):
    # compute more than just one metrics

    chosen_metrics = {
        'f1': metrics.f1_score,
        'precision': metrics.precision_score,
        'recall': metrics.recall_score,
        'accuracy': metrics.accuracy_score,
        'auc' : metrics.roc_auc_score,
    }
    results = {}
    for metric_name, metric_func in chosen_metrics.items():
        try:
            if metric_name in ["f1", "precision", "recall"]:
                score = metric_func(y_true, y_pred, average=average)
            else:
                score = metric_func(y_true, y_pred)
        except Exception as e:
            score = None
            logging.exception("Couldn't evaluate %s. %s", metric_name, e)
        results[metric_name] = score

    return results

def extract_clf_name(clf):
    # if isinstance(clf, imbPipeline):
    #     pipe_name = []
    #     for (name, step) in clf.steps:
    #         if not 'SelectPercentile' in str(step):
    #             method = (str(step).split('(')[0].lower())
    #         else:
    #             method = ('' + str(step.score_func.func_name).split('(')[0].lower())
    #         stepnamed = '{}{}'.format(name.lower(), method.upper())
    #         pipe_name.append(stepnamed)
    #     pipe_name = '+'.join(pipe_name)
    pipe_name = str(type(clf)).split(".")[-1]
    return pipe_name

def my_scorer(clf, X_val, y_true_val):

    log_name = extract_clf_name(clf)
    metric = settings.SCORER_METRIC

    # do all the work and return some of the metrics
    y_pred_val = clf.predict(X_val)

    y_pred_val[y_pred_val >= 0.5] = 1
    y_pred_val[y_pred_val < 0.5] = 0

    results = get_metrics(y_true=y_true_val, y_pred=y_pred_val)

    try:
        clf_params = clf.get_params()
    except TypeError as e:
        logging.info(e)
        clf_params = clf.sk_params
    logging.info("IN CVSEARCH CLF: {} {}"
                 "\nIN CVSEARCH RESULTS: {}".format(log_name, clf_params, results))

    if settings.SCORER_FOLD_MODEL_DIRP:
        os.makedirs(fold_model_dirp, exist_ok=True)
        model_save_fp = save_model(clf, fold_model_dirp, log_name=log_name)
    else:
        model_save_fp = None

    if settings.SCORER_FOLD_LOG_DIRP:
        os.makedirs(settings.SCORER_FOLD_LOG_DIRP, exist_ok=True)
        log_metrics_and_params(clf, results, settings.SCORER_FOLD_LOG_DIRP, y_true=y_true_val, y_pred=y_pred_val,
                               model_save_fp=model_save_fp, log_name=log_name)

    return results[metric]