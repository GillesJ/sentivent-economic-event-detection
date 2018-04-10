#!/usr/bin/env python3
'''
eval_wordvectors
sentifmdetect17 
12/18/17
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import pandas
from glob import glob
import os
from glove import Glove
from glove import metrics
from collections import defaultdict
import multiprocessing
import numpy as np
import operator
import math

def eval_glove(glove_fp):
    # Load the GloVe model
    if "stanford" in glove_fp:
        glove = Glove.load_stanford(glove_fp)
    else:
        glove = Glove.load(glove_fp)


    encode = lambda words: [str(x.lower()) for x in words]

    # Load the analogy task dataset. One example can be obtained at
    # https://word2vec.googlecode.com/svn/trunk/questions-words.txt
    sections = defaultdict(list)
    evaluation_words = [sections[section].append(encode(words)) for section, words in
                        metrics.read_analogy_file(eval_file)]

    section_ranks = []

    for section, words in sections.items():
        evaluation_ids = metrics.construct_analogy_test_set(words,
                                                            glove.dictionary,
                                                            ignore_missing=True)

        # Get the rank array.
        ranks = metrics.analogy_rank_score(evaluation_ids, glove.word_vectors,
                                           no_threads=int(njobs))
        section_ranks.append(ranks)

        print('Section %s mean rank: %s, accuracy: %s' % (section, ranks.mean(),
                                                          (ranks == 0).sum() / float(len(ranks))))

    ranks = np.hstack(section_ranks)

    rank = ranks.mean()
    accuracy = (ranks == 0).sum() / float(len(ranks))

    print('Overall rank: %s, accuracy: %s' % (rank,
                                              accuracy))

    return rank, accuracy

if __name__ == "__main__":

    eval_file = "./static/sentifmdetect/word2vec-google-eval-questions-words.txt"
    njobs = math.floor(multiprocessing.cpu_count()/4)

    scores = {}

    # glove_fps = glob(".//sentifmdetect/output/en_maintype_2017-12-18_16-18-44_WORDVECTORS/*.glovemodel") # EN_MAINTYPE TRAINED

    glove_fps = glob("./stanfordpretraineddir/*/*d.txt") # STANFORD TRAINEDs

    for i, glove_fp in enumerate(glove_fps):

        print("====={}/{}===========================\nEvaluating {}".format(
            i+1,
            len(glove_fps),
            os.path.basename(glove_fp).upper()))

        rank, accuracy = eval_glove(glove_fp)

        scores[os.path.basename(glove_fp)] = (rank, accuracy)

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1)) # sort by rank lower is better 0.5 rank is chance

    print("dataset_name\trank\taccuracy")
    for score in sorted_scores:
        print("{}\t{}\t{}".format(score[0], score[1][0], score[1][1]))



