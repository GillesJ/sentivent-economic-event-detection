#!/usr/bin/env python3
'''
wordvectors_train.py
sentifmdetect17 
12/18/17
Copyright (c) Gilles Jacobs. All rights reserved.

Trains word vectors on our corpus
'''
from sentifmdetect import settings
from sentifmdetect import util
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import numpy as np
import os
import json
import pprint
from glove import Glove
from glove import Corpus
import logging

util.setup_logging()
np.random.seed(settings.RANDOM_SEED)

def train_glove(inst, meta_data={}):

    start_total = datetime.now()

    meta_data["glove_params"] = settings.GLOVE_PARAMS

    glove_paramgrid = ParameterGrid(settings.GLOVE_PARAMS)

    for params in glove_paramgrid:

        start = datetime.now()
        # MAKE CORPUS
        # set corpus filepath
        corpus_fp = os.path.join(settings.WVEC_OPT_DIRP, '{}_window{}.glovecorpus'.format(
            settings.DATASET,
            params["window"]))
        # load if corpus exists
        if os.path.isfile(corpus_fp):
            logging.info("Loading existing corpus {}.".format(corpus_fp))
            corpus_model = Corpus.load(corpus_fp)
            logging.info("Successfully loaded existing corpus {}.".format(corpus_fp))
        # make a new coocurrence corpus if it does not exist
        else:
            logging.info("Creating new corpus at {}.".format(corpus_fp))
            corpus_model = Corpus()
            corpus_model.fit(inst, window=params["window"])
            os.makedirs(settings.WVEC_OPT_DIRP, exist_ok=True)
            corpus_model.save(corpus_fp)

        logging.info("Dict size: {}.".format(len(corpus_model.dictionary)))
        logging.info("Collocations: {}.".format(corpus_model.matrix.nnz))

        # GLOVE VECTOR TRAINING
        glove = Glove(no_components=params["dims"], learning_rate=params["lr"])

        logging.info("Start fitting GloVe with parameters: {}.".format(params))
        glove.fit(corpus_model.matrix, epochs=params["epochs"],
                  no_threads=params["njobs"], verbose=False)
        glove.add_dictionary(corpus_model.dictionary)

        os.makedirs(settings.WVEC_OPT_DIRP, exist_ok=True)
        model_name = 'glove.{}_w{}_lr{}_ep{}.{}d.glovemodel'.format(settings.DATASET,
                                                                    params["window"],
                                                                    params["lr"],
                                                                    params["epochs"],
                                                                    params["dims"])
        glove.save(os.path.join(settings.WVEC_OPT_DIRP, model_name))

        duration = (datetime.now() - start).total_seconds()
        meta_data["models"][model_name] = params
        meta_data["models"][model_name]["duration_training"] = duration

        logging.info("Finished fitting GloVe {} in {}s with parameters: {}.".format(
            model_name,
            duration,
            params))
        # SIMILARITY TEST
        for test_word in settings.TESTSIM_WORDS:
            if test_word not in meta_data["most_similar"]:
                meta_data["most_similar"][test_word] = {}

            logging.info("Querying model {} for {} most similar to \'{}\':".format(
                model_name,
                settings.N_TESTSIM,
                test_word))
            sim = glove.most_similar(test_word, number=settings.N_TESTSIM)
            meta_data["most_similar"][test_word][model_name] = sim

            logging.info(pprint.pformat(sim))

    total_duration = (datetime.now() - start_total).total_seconds()
    meta_data["glove_duration_training"] = total_duration

    return meta_data

def main():

    start_total = datetime.now()

    # DATA LOADING
    with open(settings.EXPERIMENT_DATA, "rt") as exp_in_test:
        experiment = json.load(exp_in_test)

    instances_in = experiment["holdin_instances"]
    inst = [settings.TOKENIZE_FUNC(inst) for inst in instances_in]

    # initiate meta_data
    meta_data = {
        "date_creation": settings.TIMESTAMP,
        "corpus": settings.DATASET,
        "corpus_fp": settings.DATA_DIR,
        "tokenize_func": str(settings.TOKENIZE_FUNC),
        "output_dir": settings.WVEC_OPT_DIRP,
        "host": settings.HOST,
        "models": {},
        "most_similar": {},
        }

    # GLOVE TRAIN
    meta_data = train_glove(inst, meta_data=meta_data)

    # WRITE META_DATA
    meta_fp = os.path.join(settings.WVEC_OPT_DIRP, "info.json")
    try:
        with open(meta_fp, "wt") as meta_out:
            json.dump(meta_data, meta_out)
        logging.info("Meta data written to {}.".format(meta_fp))
        # # # test meta_data
        # with open(meta_fp, "rt") as meta_in:
        #     meta_test = json.load(meta_in)
        # pprint.pprint(meta_test)
    except Exception as e:
        logging.info("Failed to write glove run metadata.", e)

    total_duration = (datetime.now() - start_total).total_seconds()
    logging.info(
        "Finished training wordvector models in {}s! Run data written to {}. Do not forget to test with the glove-python packages included analogy_task_evaluation.py script".format(
            total_duration,
            settings.WVEC_OPT_DIRP))


if __name__ == "__main__":
    main()