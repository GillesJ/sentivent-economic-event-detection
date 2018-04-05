#!/usr/bin/env python3
'''
classifier.py
sentifmdetect17 
11/24/17
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
from sentifmdetect import settings
from sentifmdetect import featurizer
from sentifmdetect import util
import os
import keras
from keras.optimizers import Adam
from keras import backend
from keras.layers import Dense, Input, Flatten, Dropout, Merge, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Bidirectional
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score, precision_score,\
    recall_score, roc_auc_score
import numpy as np

# util.setup_logging()

def create_emb_lstm(
        wvec=50, # can be the filepath as string of a pretrained wordvector in settings.ALL_EMB_FP or an int for no pretrained specifying dimensionality
        bidirectional=False,
        lstm_units=10,
        lstm_dropout=0.2,
        lstm_recurrent_dropout=0.2,
        optimizer=(Adam, {}),
        metrics=["accuracy"]):

    # Make the model add kwargs for each layer
    model = Sequential()

    if os.path.isfile(wvec): # if wvec is a path, pretrained word vectors are available
        # Load pretrained embeddings
        embeddings_index = featurizer.load_emb(wvec)
        featurizer.compute_pretrained_coverage(settings.WORD_INDEX, embeddings_index) # for log
        # make emb matrix
        EMBEDDINGS_MATRIX = featurizer.make_embedding_matrix(settings.WORD_INDEX, embeddings_index)
        EMB_DIM = EMBEDDINGS_MATRIX.shape[1]
        model.add(
            Embedding(settings.EMB_INPUT_DIM, EMB_DIM, weights=[EMBEDDINGS_MATRIX], input_length=settings.MAX_SEQUENCE_LENGTH))
    elif isinstance(wvec, int):
        EMB_DIM = wvec
        model.add(
            Embedding(settings.EMB_INPUT_DIM, EMB_DIM, input_length=settings.MAX_SEQUENCE_LENGTH))
    else:
        logging.error("NO EMBEDDINGS ARE GIVEN.")

    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)))
    else:
        model.add(LSTM(lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))

    model.add(Dense(settings.OUTPUT_UNITS, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer[0](**optimizer[1]), metrics=metrics)

    return model

class KerasClassifierCustom(KerasClassifier):
    '''
    Override the KerasClassifier predict() method to use the predict() function instead of predict_classes()
    function which cannot be scored.
    '''
    def predict(self, x, **kwargs):
        """Returns the predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        return self.model.predict(x, **kwargs)

class GlobalMetrics(keras.callbacks.Callback):
    '''
    This class facilitates the use of global performance metric such as Precision, Recall, Fscore, etc.
    Global metrics should not be calculated during training on batches, because by definition they are meant
    to assess the performance of a classifier on the full set. Hence they are evaluated on the validation set at epoch
    end.
    To be used in the callbacks parameter of a model.fit() call.
    '''
    def __init__(self, metrics, from_categorical=True):
        self.from_categorical = True

        if isinstance(metrics, list):
            self.global_metrics = metrics
        else:
            raise TypeError("The metrics argument should be a list of tuples (metric functions, kwargs).")

        self.global_scores = {}


    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]

        if self.from_categorical:
            predict = predict.argmax(axis=-1)
            targ = targ.argmax(axis=-1)

        for metric, kwargs in self.global_metrics:
            self.global_scores[metric.__name__] = metric(targ, predict, **kwargs)
            print("\nval_{}: {}".format(metric.__name__, self.global_scores[metric.__name__]))
        return

class KerasRandomizedSearchCV(RandomizedSearchCV):
    '''
    Provides SearchCV for use with Keras classifiers. Overrides the predict method in order to clear the session and free
    GPU memory.
    '''

    # def fit(self, *args, **kwargs):
    #     super(KerasRandomizedSearchCV, self).fit(*args, **kwargs)
    #     return self

    def predict(self, *args, **kwargs):
        pred = super(KerasRandomizedSearchCV, self).predict(*args, **kwargs)
        backend.clear_session()
        return pred



if __name__ == "__main__":

    # some testing
    from sklearn.datasets import make_moons
    from sklearn.model_selection import RandomizedSearchCV
    from keras.regularizers import l2

    dataset = make_moons(1000)

    def build_fn(nr_of_layers=2,
                 first_layer_size=10,
                 layers_slope_coeff=0.8,
                 dropout=0.5,
                 activation="relu",
                 weight_l2=0.01,
                 act_l2=0.01,
                 input_dim=2):

        result_model = Sequential()
        result_model.add(Dense(first_layer_size,
                               input_dim=input_dim,
                               activation=activation,
                               W_regularizer=l2(weight_l2),
                               activity_regularizer=l2(act_l2)
                               ))

        current_layer_size = int(first_layer_size * layers_slope_coeff) + 1

        for index_of_layer in range(nr_of_layers - 1):
            result_model.add(BatchNormalization())
            result_model.add(Dropout(dropout))
            result_model.add(Dense(current_layer_size,
                                   W_regularizer=l2(weight_l2),
                                   activation=activation,
                                   activity_regularizer=l2(act_l2)
                                   ))

            current_layer_size = int(current_layer_size * layers_slope_coeff) + 1

        result_model.add(Dense(1,
                               activation="sigmoid",
                               W_regularizer=l2(weight_l2)))

        result_model.compile(optimizer="rmsprop", metrics=["accuracy"], loss="binary_crossentropy")

        return result_model


    NeuralNet = KerasClassifier(create_pretrained_emb_lstm)