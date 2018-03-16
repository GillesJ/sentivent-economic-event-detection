#!/usr/bin/env python3
'''
settings.py
sentifmdetect17 
11/24/17
Copyright (c) Gilles Jacobs. All rights reserved.

Setting constants for the experiments.
'''
import numpy as np
import socket
import multiprocessing
import keras
import math
from sentifmdetect.classifier import GlobalMetrics
from sentifmdetect.featurize import text_to_word_sequence_nltkword, text_to_word_sequence_stanford
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from glob import glob
import os
from datetime import datetime

# RANDOM SEED
RANDOM_SEED = 92 # set this in the experiment code with np.random.seed(), CAVEAT: np.random.seed is not thread-safe:
# sklearn multithreaded objects have a random_state argument to which you have to pass the seed.
np.random.seed(RANDOM_SEED)

# DATA FILES
LANGUAGE = "en"
DATASET = "{}_maintype".format(LANGUAGE)
EXPERIMENT_DATA = "/home/gilles/repos/sentifmdetect17/sentifmdetect/static/experiment_data.json" # made for sharing
ALL_DATA_DIR = {
    "en_maintype": "/home/gilles/corpora/sentifm/data/English/MainType",
    "nl_maintype": "/home/gilles/corpora/sentifm/data/Dutch/MainType",
}
ALL_EXCLUDE = {
    "en_maintype": [],
    "nl_maintype": ["/home/gilles/corpora/sentifm/data/Dutch/MainType/BuyRating_nl_train_uniq.txt"],
}
EXCLUDE = ALL_EXCLUDE[DATASET]
DATA_DIR = ALL_DATA_DIR[DATASET]

# METADATA
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
HOST = socket.gethostname()

# OUTPUT
OPT_DIRP = f"/home/gilles/repos/sentifmdetect17/output/{DATASET}_{TIMESTAMP}" # TODO REMEMBER THIS WAS RELATIVE PATH AND DIFFERS WHEN CALLED IN MODULE DO NOT DELETE ANY OUTPUT DIR
SCORER_FOLD_LOG_DIRP = os.path.join(OPT_DIRP, 'fold_log')
SCORER_FOLD_MODEL_DIRP = None
# SCORER_FOLD_MODEL_DIRP = os.path.join(OPT_DIRP, 'fold_models')

SCORER_METRIC = 'f1'
SCORE_AVERAGING = 'macro'

# EXPERIMENT SETTINGS
HOLDOUT_SPLIT = 0.1
CV_SPLIT = 0.1
KFOLD = 3

KERAS_METRICS = GlobalMetrics([(f1_score, {}),
                        (precision_score, {}),
                        (recall_score, {}),
                        (roc_auc_score, {})],
                        from_categorical=True)
# FEATURIZE
# EMB_DIM = 200
# # for sentifm trained NAACL paper
# EMB_NAME = "glove.en_maintype_w15_lr0.25_ep20.{}d.glovemodel".format(EMB_DIM)  # TODO make embedding file part of hyperparam search optim
# # for stanford glove.6b 3fold NAACL conf paper
# EMB_NAME = "glove.6B.300d".format(EMB_DIM)

EMB_FP = {
    "glove.6B.50d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.50d.txt",
    "glove.6B.100d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.100d.txt",
    # "glove.6B.200d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.200d.txt",
    # "glove.6B.300d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.300d.txt",
    # "glove.42B.300d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.42B.300d/glove.42B.300d.txt",
    # "glove.840B.300d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.840B.300d/glove.840B.300d.txt",
    "glove.twitter.27B.25d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.twitter.27B/glove.twitter.27B.25d.txt",
    "glove.twitter.27B.50d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.twitter.27B/glove.twitter.27B.50d.txt",
    "glove.twitter.27B.100d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.twitter.27B/glove.twitter.27B.100d.txt",
    "glove.twitter.27B.200d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.twitter.27B/glove.twitter.27B.200d.txt",
	"glove.en_maintype_w15_lr0.25_ep20.50d.glovemodel": # own trained model top 10 best on google word2vec analogy eval (cf. /static)
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep20.50d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep30.100d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep30.100d.glovemodel",
	"glove.en_maintype_w10_lr0.25_ep20.100d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w10_lr0.25_ep20.100d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep50.50d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep50.50d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep20.25d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep20.25d.glovemodel",
	"glove.en_maintype_w10_lr0.25_ep20.50d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w10_lr0.25_ep20.50d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep20.200d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep20.200d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep30.25d.glovemodel": # seem to do well
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep30.25d.glovemodel",
	"glove.en_maintype_w10_lr0.25_ep30.100d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w10_lr0.25_ep30.100d.glovemodel",
	"glove.en_maintype_w10_lr0.25_ep30.25d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/glove.en_maintype_w10_lr0.25_ep30.25d.glovemodel",
}
# for settings the tokenization function
TOKENIZERS = {
    "en": keras.preprocessing.text.text_to_word_sequence, # default tokenize func of keras.Tokenizer: glove.6B coverage 89.78% BEST
    # "en": text_to_word_sequence_nltkword, # TODO setup CORENNLP because is used for glove.6B
    # "en": text_to_word_sequence_stanford, # nltk tokenizer, 82.09% coverage of glove.6B stanford emb
}
TOKENIZE_FUNC = TOKENIZERS[LANGUAGE]# func to monkeypatch the keras tokenizer that matches the pretrained embeddings

# GLOVE SETTINGS
WVEC_OPT_DIRP = OPT_DIRP + "_WORDVECTORS"
TESTSIM_WORDS = ["company", "profit", "merger", "rise", "price", "london", "dividend", "margin", "recover", "debt", "ipo"]
N_TESTSIM = 10 # number of sim results to retrieve for test
GLOVE_PARAMS = {
    "njobs": [math.floor(multiprocessing.cpu_count()/2)], # parallelism; threads to use
    "window": [5, 10, 15], # word window for cooccurence calculation
    "lr": [0.05, 0.25, 0.01], # glove learning rate
    "epochs": [20, 30, 50], # epochs in training
    "dims": [25, 50, 100, 200, 300, 400, 500, 600], # no
}

