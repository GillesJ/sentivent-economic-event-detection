#!/usr/bin/env python3
'''
crossvalidate.py
sentifmdetect17 
11/24/17
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
from sentifmdetect import datahandler
from sentifmdetect import settings
from sentifmdetect import scorer
from sentifmdetect import featurize
from sentifmdetect import util
from sentifmdetect import classifier
import os
from sklearn.externals import joblib
import logging
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import numpy as np
from keras.optimizers import Adam
import json
from keras.models import save_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Flatten, Dropout, Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Bidirectional
from keras.models import Model, Sequential
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score, precision_score,\
    recall_score, roc_auc_score
from keras import backend

util.setup_logging()

# TODO keep experiment data metadata etc in 1 json file per run
# TODO test early stopping callback, if issue cf https://github.com/keras-team/keras/issues/4278
# TODO add decent logging
# TODO check what is wrong with the OPT_DIR
# TODO add checkpointing: dumping and reloading
# TODO add pandas result collection
# TODO add learning curves
# TODO find out if it is necessary or not to set np.random_seed for each import

if __name__ == "__main__":

    np.random.seed(settings.RANDOM_SEED)

    # global EMB_INPUT_DIM, EMBEDDINGS_MATRIX, MAX_SEQUENCE_LENGTH, OUTPUT_UNITS
    logging.info("Start crossvalidation binary task and test with output dir {}".format(settings.OPT_DIRP))
    # Load data
    instances, labels = datahandler.load_corpus(settings.DATA_DIR)

    all_stats = datahandler.get_label_stats(labels) # for paper

    print(sum([v for k, v in all_stats["multilabel_distr"].items() if "," in k]))
    print(sum([v for k, v in all_stats["multilabel_count"].items() if "," in k]))

    # Encode labels
    labels_orig = np.array(labels, dtype=object)
    labels = [l if l != -1 else [] for l in labels] # TODO test with negative label instances as a label in learning
    label_encoder = MultiLabelBinarizer() # makes multihot label encodings
    y = label_encoder.fit_transform(labels)

    # Make sequence data from text
    # # Load predetermined holdout split
    with open(settings.EXPERIMENT_DATA, "rt") as exp_in_test:
        experiment = json.load(exp_in_test)

    idc_in = experiment["meta_holdin_indices"]
    idc_out = experiment["meta_holdout_indices"]
    # when code testing limit the instances by using the slicing done by predetermined holdout split indices
    if settings.TEST:
        idc_in, idc_out = idc_in[:40], idc_out[:10]

    x, word_index, settings.MAX_SEQUENCE_LENGTH = featurize.make_sequences(instances)

    x_in = x[idc_in]
    x_out = x[idc_out]
    y_in = y[idc_in]
    y_out = y[idc_out]
    labels_in = labels_orig[idc_in]
    labels_out = labels_orig[idc_out]

    logging.info("Train class category counts: \n{}\n---------\n"
          "Test class category counts: \n{}.".format(datahandler.get_label_info(labels_in),
                                                     datahandler.get_label_info(labels_out)))

    # set global vars
    settings.EMB_INPUT_DIM = len(word_index) + 1
    settings.OUTPUT_UNITS = len(label_encoder.classes_)
    settings.WORD_INDEX = word_index

    # SKLEARN WRAPPER
    multilab_lstm = classifier.KerasClassifierCustom(classifier.create_emb_lstm)

    param_grid = {
        "wvec": [25, 50, 100, 300] + list(settings.EMB_FP.values()),
        "bidirectional": [True, False],
        "lstm_units": [int(settings.MAX_SEQUENCE_LENGTH * 8), int(settings.MAX_SEQUENCE_LENGTH * 16), int(settings.MAX_SEQUENCE_LENGTH * 32)],
        "lstm_dropout": [0.0, 0.2],
        "lstm_recurrent_dropout": [0.0, 0.2],
        "optimizer": [
            (Adam, {"lr":0.001, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":0.0}), # we do not init any objects in params so that they are visible as dict to the CV.get_params().
            # (Adam, {"lr":0.002, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":1.0}), # this does not seem to do better, ever
        ],
        "batch_size": [64, 128, 256,],
        "epochs": [32, 64],
    }

    # searchcv = classifier.KerasRandomizedSearchCV(
    searchcv = RandomizedSearchCV(
        multilab_lstm,
        param_distributions=param_grid,
        n_iter=32,
        cv=settings.KFOLD,
        scoring=scorer.my_scorer,
        fit_params=dict(callbacks=[settings.EARLY_STOP, ]), # hack for adding keras callbacks
        verbose=0,
        error_score=0, # value 0 ignores failed fits and move on to next fold
        return_train_score=False, # circumvents bug in sklearn, prevent return the meaningless train score which will always be near perfect
        random_state=settings.RANDOM_SEED,
        n_jobs=1,
        )

    if settings.TEST:
        searchcv = RandomizedSearchCV(
            multilab_lstm,
            param_distributions=settings.PARAM_GRID_TEST,
            n_iter=1,
            cv=2,
            scoring=scorer.my_scorer,
            fit_params=dict(callbacks=[early_stop, ]),
            verbose=0,
            error_score=0,
            return_train_score=False,
            random_state=settings.RANDOM_SEED,
            n_jobs=1,
        )

    # searchcv = EvolutionaryAlgorithmSearchCV( # TODO experiment with Evolutionary Search
    #     multilab_lstm,
    #     params=param_grid,
    #     cv=StratifiedKFold(n_splits=settings.KFOLD),
    #     scoring=scorer.my_scorer,
    #     population_size=64,
    #     gene_mutation_prob=0.3,
    #     gene_crossover_prob=0.5,
    #     tournament_size=3,
    #     generations_number=8,
    #     verbose=2,
    #            )

    # MULTILABEL TO BINARY TASKS:
    for bin_label in np.unique(labels_in):
        logging.info(f"Binary task {bin_label}: running cv optimization.")

        # transform output label
        pos_label = label_encoder.transform(bin_label)
        y_in = (y_in == pos_label).astype(int)
        y_out = (y_in == pos_label).astype(int)

        searchcv.fit(x_in, y_in, verbose=2)

        try:
            best_params = searchcv.best_params_
        except TypeError as e:
            logging.info(e)
            best_params = searchcv.best_estimator_.sk_params

        logging.info("CV BEST ESTIMATOR: {}".format(searchcv.best_estimator_))
        logging.info("CV BEST PARAMETERS: {}".format(best_params))
        logging.info("CV BEST SCORE: {}".format(searchcv.best_score_))

        # try:
        #     os.makedirs(settings.OPT_DIRP, exist_ok=True)
        #     cv_fp = os.path.join(settings.OPT_DIRP, "cv_param_search.joblibpickle")
        #     joblib.dump(searchcv, cv_fp)
        # except Exception as e:
        #     logging.exception("Failed to pickle search object. ", e) # write code to do this not supported by default

        y_pred = searchcv.predict(x_out)

        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        # multilabel_pred = np.unique(np.count_nonzero(y_pred, axis=1), return_counts=True) # code to check whether multilabel multihots are in deed predicted
        # logging.info(multilabel_pred)

        clf_report = classification_report(y_out, y_pred, target_names=label_encoder.classes_) # TODO the avg/total does not take into account labels

        try:
            os.makedirs(settings.OPT_DIRP, exist_ok=True)
            with open(os.path.join(settings.OPT_DIRP, "holdout_report.txt"), "wt") as report_out:
                report_out.write(clf_report)
        except Exception as e:
            logging.exception("Failed to write classification report.", e)

        logging.info(clf_report)