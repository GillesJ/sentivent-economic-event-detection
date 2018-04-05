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
from keras.callbacks import EarlyStopping
import math
from sklearn.model_selection import StratifiedKFold
from sentifmdetect.classifier import GlobalMetrics
from sentifmdetect.featurizer import text_to_word_sequence_nltkword, text_to_word_sequence_stanford, corenlp_tokenize_enpbt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from glob import glob
import os
from datetime import datetime
from pytz import timezone
import corenlp

#TESTING enable to limit the amount of trained and tested instances for a test run
TEST = False
N_TEST_INSTANCES = 200
# RANDOM SEED
RANDOM_SEED = 92 # set this in the experiment code with np.random.seed(), CAVEAT: np.random.seed is not thread-safe:
# sklearn multithreaded objects have a random_state argument to which you have to pass the seed.
np.random.seed(RANDOM_SEED)

# DATA FILES
LANGUAGE = "en"
TASK = "maintype"
DATASET = f"{LANGUAGE}_{TASK}" if not TEST else f"TEST{N_TEST_INSTANCES}_{LANGUAGE}_{TASK}"
EXPERIMENT_DATA = "/home/gilles/repos/sentifmdetect17/sentifmdetect/static/experiment_data.json" # made for sharing
ALL_DATA_DIR = {
    "en_maintype": "/home/gilles/corpora/sentifm/data/English/MainType",
    "nl_maintype": "/home/gilles/corpora/sentifm/data/Dutch/MainType",
}
ALL_EXCLUDE = { # data path to skip for inclusion because it redundant and badly parsed
    "en_maintype": [],
    "nl_maintype": ["/home/gilles/corpora/sentifm/data/Dutch/MainType/BuyRating_nl_train_uniq.txt"],
}
EXCLUDE = ALL_EXCLUDE[f"{LANGUAGE}_{TASK}"]
DATA_DIR = ALL_DATA_DIR[f"{LANGUAGE}_{TASK}"]
FEATURE_OPT_FP = f"/home/gilles/repos/sentifmdetect17/features/{DATASET}_features.json"

# OUTPUT
TIMESTAMP = datetime.now(timezone("Europe/Brussels")).strftime("%Y-%m-%d_%H:%M:%S_%Z")
OPT_DIRP = f"/home/gilles/repos/sentifmdetect17/output/{DATASET}_{TIMESTAMP}"
os.makedirs(OPT_DIRP, exist_ok=True)
SCORER_FOLD_LOG_DIRP = os.path.join(OPT_DIRP, 'fold_log')
SCORER_FOLD_MODEL_DIRP = None
# SCORER_FOLD_MODEL_DIRP = os.path.join(OPT_DIRP, 'fold_models')

SCORER_METRIC = 'f1'
SCORE_AVERAGING = 'macro'

# EXPERIMENT SETTINGS
HOLDOUT_SPLIT = 0.1
CV_SPLIT = 0.1
KFOLD = StratifiedKFold(n_splits=3, random_state=np.random.seed(RANDOM_SEED))

VAL_METRICS = GlobalMetrics([(f1_score, {}),
                        (precision_score, {}),
                        (recall_score, {}),
                        (roc_auc_score, {})],
                        from_categorical=True)

# Keras callbacks
EARLY_STOP = EarlyStopping(
    monitor='val_acc',  # early stopping on accuracy (maybe bad idea), generous patience of 5
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto')

# FEATURIZE
# EMB_DIM = 200
# # for sentifm trained NAACL paper
# EMB_NAME = "glove.en_maintype_w15_lr0.25_ep20.{}d.glovemodel".format(EMB_DIM)  # TODO make embedding file part of hyperparam search optim
# # for stanford glove.6b 3fold NAACL conf paper
# EMB_NAME = "glove.6B.300d".format(EMB_DIM)

EMB_FP = {
    "glove.6B.50d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.50d.txt",
    "glove.6B.100d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.100d.txt",
    "glove.6B.200d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.200d.txt",
    "glove.6B.300d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.300d.txt",
    "glove.42B.300d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.42B.300d/glove.42B.300d.txt",
    "glove.840B.300d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.840B.300d/glove.840B.300d.txt",
    "glove.twitter.27B.25d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.twitter.27B/glove.twitter.27B.25d.txt",
    "glove.twitter.27B.50d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.twitter.27B/glove.twitter.27B.50d.txt",
    "glove.twitter.27B.100d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.twitter.27B/glove.twitter.27B.100d.txt",
    "glove.twitter.27B.200d": "/home/gilles/corpora/word_embeddings/glove_stanford/glove.twitter.27B/glove.twitter.27B.200d.txt",
	"glove.en_maintype_w15_lr0.25_ep20.50d.glovemodel": # own trained model top 10 best on google word2vec analogy eval (cf. /static)
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep20.50d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep30.100d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep30.100d.glovemodel",
	"glove.en_maintype_w10_lr0.25_ep20.100d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w10_lr0.25_ep20.100d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep50.50d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep50.50d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep20.25d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep20.25d.glovemodel",
	"glove.en_maintype_w10_lr0.25_ep20.50d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w10_lr0.25_ep20.50d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep20.200d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep20.200d.glovemodel",
	"glove.en_maintype_w15_lr0.25_ep30.25d.glovemodel": # seem to do well
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w15_lr0.25_ep30.25d.glovemodel",
	"glove.en_maintype_w10_lr0.25_ep30.100d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w10_lr0.25_ep30.100d.glovemodel",
	"glove.en_maintype_w10_lr0.25_ep30.25d.glovemodel":
		"/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-22_18:41:52_CET_WORDVECTORS/glove.en_maintype_w10_lr0.25_ep30.25d.glovemodel",
}
# for settings the tokenization function
TOKENIZERS = {
    "en": corenlp_tokenize_enpbt
    # "en": keras.preprocessing.text.text_to_word_sequence, # default tokenize func of keras.Tokenizer: glove.6B coverage 89.78% BEST
    # "en": text_to_word_sequence_nltkword, # TODO setup CORENNLP because is used for glove.6B
    # "en": text_to_word_sequence_stanford, # nltk tokenizer, 82.09% coverage of glove.6B stanford emb
}
TOKENIZE_FUNC = TOKENIZERS[LANGUAGE]# func to monkeypatch the keras tokenizer that matches the pretrained embeddings
if TOKENIZE_FUNC in [corenlp_tokenize_enpbt]:
    CORENLP_CLIENT = corenlp.CoreNLPClient(annotators=["tokenize"])

# small param grid used when code testing
PARAM_GRID_TEST = {
    "wvec": [EMB_FP["glove.en_maintype_w15_lr0.25_ep20.50d.glovemodel"]], # for testing self vectors + init with first of pretrained
    # "wvec": [EMB_FP["glove.6B.50d"]],
    "bidirectional": [False],
    "lstm_units": [64],
    "batch_size": [128],
    "epochs": [32],
}
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

# METADATA init
METADATA = {
    "random_seed": RANDOM_SEED,
    "host": socket.gethostname(),
    "timestamp": TIMESTAMP,
}