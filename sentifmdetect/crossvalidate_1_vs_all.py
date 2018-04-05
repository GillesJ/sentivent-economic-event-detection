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
from sentifmdetect import featurizer
from sentifmdetect import util
from sentifmdetect import classifier
import os
from sklearn.externals import joblib
import logging
import numpy as np
from keras.optimizers import Adam
import json
from sklearn.utils.class_weight import compute_sample_weight
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
    premade_feature_data = util.read_features()
    if premade_feature_data:
        feature_data = premade_feature_data
    else:
        feature_data = featurizer.featurize()

    util.write_metadata({"feature_data": feature_data})

    x_in = np.array(feature_data["x_in"])
    x_out = np.array(feature_data["x_out"])
    y_in = np.array(feature_data["y_in"])
    y_out = np.array(feature_data["y_out"])
    labels_in = feature_data["labels_in"]
    labels_out = feature_data["labels_out"]
    classes = feature_data["classes"]
    # # FOR ECONLP FIX PARTIAL RUN
    # classes.remove("BuyRating")

    # set global vars
    settings.MAX_SEQUENCE_LENGTH = feature_data["max_sequence_length"]
    settings.EMB_INPUT_DIM = feature_data["emb_input_dim"]
    # settings.OUTPUT_UNITS = feature_data["output_units"] # override for binary emb_lstm
    settings.OUTPUT_UNITS = 1
    settings.WORD_INDEX = feature_data["word_index"]

    # SKLEARN WRAPPER
    clf = classifier.KerasClassifierCustom(classifier.create_emb_lstm)

    param_grid = {
        "wvec": [settings.EMB_FP["glove.6B.300d"], settings.EMB_FP["glove.en_maintype_w15_lr0.25_ep20.200d.glovemodel"]],
        "bidirectional": [True, False],
        "lstm_units": [int(settings.MAX_SEQUENCE_LENGTH * 4), int(settings.MAX_SEQUENCE_LENGTH * 8), int(settings.MAX_SEQUENCE_LENGTH * 16)],
        "lstm_dropout": [0.0, 0.2],
        "lstm_recurrent_dropout": [0.0, 0.2],
        "optimizer": [
            (Adam, {"lr":0.001, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":0.0}), # we do not init any objects in params so that they are visible as dict to the CV.get_params().
            # (Adam, {"lr":0.002, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":1.0}), # this does not seem to do better, ever
        ],
        "batch_size": [64, 128, 256],
        "epochs": [32],
    }


    # MULTILABEL TO 1 vs ALL TASKS:
    for bin_label in classes:
        logging.info(f"Binary task {bin_label}: running cv optimization.")
        settings.POS_LABEL = bin_label
        settings.CURR_FOLD = 0

        y_in = np.array([-1 if lab == -1 else 1 if bin_label in lab else -1 for lab in labels_in])
        y_out = np.array([-1 if lab == -1 else 1 if bin_label in lab else -1 for lab in labels_out])
        val_sample_weight = compute_sample_weight("balanced", y_out) # for validation set balanced accuracy

        #TODO add sample weights to fit method
        searchcv = classifier.KerasRandomizedSearchCV(
            clf,
            param_distributions=param_grid,
            n_iter=32,
            cv=settings.KFOLD,
            scoring=scorer.my_scorer,
            verbose=0,
            error_score=0,  # value 0 ignores failed fits and move on to next fold
            return_train_score=False, # circumvent bug sklearn, prevent return of near-perfect meaningless train score
            random_state=settings.RANDOM_SEED,
            n_jobs=1,
        )

        if settings.TEST:
            searchcv = classifier.KerasRandomizedSearchCV(
                clf,
                param_distributions=settings.PARAM_GRID_TEST,
                n_iter=1,
                cv=settings.KFOLD,
                scoring=scorer.my_scorer,
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

        # transform output label

        searchcv.fit(x_in, y_in, validation_data=(x_out, y_out), callbacks=[settings.EARLY_STOP, ], verbose=2)

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

        y_pred = y_pred.flatten() #flatten for binary task

        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = -1
        y_pred = y_pred.astype(int)

        scores = scorer.get_metrics(y_out, y_pred, average="binary")
        logging.info(f"{bin_label}: {scores}")

        # add to metadata
        util.write_metadata({"binary_task": {
            bin_label: {
                "y_in": y_in.tolist(),
                "y_out": y_out.tolist(),
                "y_pred": y_pred.tolist(),
                "best_estimator": str(searchcv.best_estimator_),
                "best_params": best_params,
                "best_cv_score": searchcv.best_score_,
                "scores": scores,
            }}})

        # try:
        #     clf_report = classification_report(y_out, y_pred, labels=[bin_label]) # TODO the avg/total does not take into account labels
        #     # clf_report_neg = classification_report(y_out, y_pred, target_names=feature_data["classes"].append(-1))
        #     try:
        #         os.makedirs(settings.OPT_DIRP, exist_ok=True)
        #         with open(os.path.join(settings.OPT_DIRP, "holdout_report.txt"), "wt") as report_out:
        #             report_out.write(clf_report)
        #     except Exception as e:
        #         logging.exception("Failed to write classification report.", e)
        #     logging.info(clf_report)
        # except Exception as e:
        #     logging.exception(e)

    logging.info(f"Completed run: metadata at {settings.OPT_DIRP}.")