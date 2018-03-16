#!/usr/bin/env python3
'''
featurize.py
sentifmdetect17 
11/24/17
Copyright (c) Gilles Jacobs. All rights reserved.
'''
from sentifmdetect import settings
from sentifmdetect import datahandler
import os
import logging
from sentifmdetect import util
import keras.preprocessing.text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.tokenize.stanford import CoreNLPTokenizer
from nltk import word_tokenize
import nltk
from glove import Glove

util.setup_logging()
nltk.download('punkt')

def text_to_word_sequence_stanford(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    # TODO INSTALL CORENLP
    """Using the Stanford Tokenizer, converts a text to a sequence of words (or tokens).
    This overrides the text_to_word_sequences method of keras.preprocessing.text.
    We monkeypatch the default tokenization method to match the tokenizer used on the pre-trained word embeddings.
    # Arguments
        text: Input text (string).
        filters: Sequence of characters to filter out. FOR COMP WITH SKLEARN
        lower: Whether to convert the input to lowercase. FOR COMP WITH SKLEARN
        split: Sentence split marker (string). FOR COMP WITH SKLEARN

    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    tokens = CoreNLPTokenizer().tokenize(text)
    return tokens

def text_to_word_sequence_nltkword(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Using the Stanford Tokenizer, converts a text to a sequence of words (or tokens).
    This overrides the text_to_word_sequences method of keras.preprocessing.text.
    We monkeypatch the default tokenization method to match the tokenizer used on the pre-trained word embeddings.
    # Arguments
        text: Input text (string).
        filters: Sequence of characters to filter out. FOR COMP WITH SKLEARN
        lower: Whether to convert the input to lowercase. FOR COMP WITH SKLEARN
        split: Sentence split marker (string). FOR COMP WITH SKLEARN

    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    tokens = word_tokenize(text)
    return tokens

def load_pretrained_emb_stanford(fp):
    embeddings_index = {}
    with open(fp, "rt") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index

def load_pretrained_emb_glovemodel(fp):
    embeddings_index = {}
    glove = Glove.load(fp)
    for word, idx in glove.dictionary.items():
        embeddings_index[word] = glove.word_vectors[idx]
    return embeddings_index

def index_embeddings(fp):
    if "glove" in fp:
        logging.info("Indexing glove pretrained word vectors.")

        if fp.endswith(".glovemodel"): # if the embeddings is glove trained with wordvectors_train.py script
            embeddings_index = load_pretrained_emb_glovemodel(fp)

        else: # stanford pretrained embeddings
            embeddings_index = load_pretrained_emb_stanford(fp)

    logging.info(f"Found {len(embeddings_index)} word vectors.")

    return embeddings_index

def load_emb(glove_fp):
    # load the glove embeddings
    emb_name = os.path.splitext(os.path.basename(glove_fp))[0]
    logging.info("Loading word vectors {}.".format(emb_name))
    embeddings_index = index_embeddings(glove_fp)
    return embeddings_index

def compute_pretrained_coverage(word_index, embeddings_index):
    # compute how well the tokenizer matches the tokens in the word embeddings file
    wordtype_overlap = 0
    not_in = []
    for wordtype in word_index.keys():
        if wordtype in embeddings_index:
            wordtype_overlap += 1
        else:
            not_in.append(wordtype)
    wordtype_overlap_pct = 100 * (wordtype_overlap / len(word_index))
    logging.info("{:.2f}% (n={}/{}) of corpus types are present in the word embeddings vector.".format(
        wordtype_overlap_pct, wordtype_overlap, len(word_index)))

def make_sequences(instances):
    # # monkeypatch the keras tokenizer to use a tokenizer that matches the pretrained embeddings
    keras.preprocessing.text.text_to_word_sequence = settings.TOKENIZE_FUNC # TODO make this work

    # Make sequences using the monkeypatched keras Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(instances)
    sequences = tokenizer.texts_to_sequences(instances)
    MAX_SEQUENCE_LENGTH = np.array([len(seq) for seq in sequences]).max()  # lets take the max instead of setting it

    logging.info(f"Longest document is {MAX_SEQUENCE_LENGTH} tokens.")

    word_index = tokenizer.word_index

    logging.info(f"Found {len(word_index)} unique tokens.")

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data, word_index, MAX_SEQUENCE_LENGTH

def make_embedding_matrix(word_index, embeddings_index):

    logging.info('Preparing embedding matrix.')
    emb_dim = next(iter(embeddings_index.values())).shape[0] # load arbitrary vector and get its length
    # prepare embedding matrix
    # num_words = min(MAX_NUM_WORDS, len(word_index))
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, emb_dim))
    for word, i in word_index.items():
        # if i >= MAX_NUM_WORDS:
        #     continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


if __name__ == "__main__":


    embeddings_indices = load_emb()
    embeddings_index = embeddings_indices["glove.6B.100d"]
    instances, _ = datahandler.load_corpus(settings.DATA_DIR)
    X = make_sequences(instances, embeddings_index)