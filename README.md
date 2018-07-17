# Economic event detection in company-specific news text
Economic event type detection on the SentiFM dataset using biLSTM and SVM for the paper: Gilles Jacobs, Els Lefever, and Véronique Hoste. 2018. Economic event detection in company-specific news text. In Proceedings of the 1st Workshop on Economics and NLP (ECONLP). ACL 2018, Melbourne, AUS, 1-10.

This repo includes data and code for company-specific sentence level event type classification for the [English SentiFM dataset](https://osf.io/enu2k/).

Please cite the original paper when using the dataset.

This code can completely replicate the experiments described in the paper with pre-processing, word-vector creation & evaluation, hyperparameter optimization in crossvalidation and holdout-prediction.

## Set-up:
1. Install non-python dependencies:
    - Install CUDA if not installed (it is already on phil)
    - `sudo apt-get install libopenblas-base libopenblas-base python-dev`
    - download and unpack latest Stanford CoreNLP: `cd ~/software; wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip ; unzip stanford-corenlp-full-2018-02-27.zip; rm stanford-corenlp-full-2018-02-27.zip` And set the envvar for the python Corenlp package to use `CORENLP_HOME=~/software/stanford-corenlp-full-2018-02-27`

2. Configure Keras to use TensorFlow:
    - set `$HOME/.keras/keras.json` to:
    ```json
    {
        "image_data_format": "channels_last",
        "epsilon": 1e-07,
        "floatx": "float32",
        "backend": "tensorflow"
    }
    ```
## Contents and usage
Set your experiment storage/output paths and experimental settings in settings.py
- `settings.py` for defining the experimental constants for crossvalidation optim. & testing, & wordvector training.
- `featurize.py` feature engineering: tokenisation, indexing, sequencing & making the embedding matrix.
- `crossvalidate.py` run validation test & multi-label crossvalidation experiment using featurized data.
- `crossvalidate.py` run validation test & one-vs-rest crossvalidation experiment using featurized data.
- `datahandler.py` loading, parsing, writing, making splits and general handling of dataset.
- `classifier.py` custom sklearn-compatible classifiers and classifier handling.
- `scorer.py` custom classifier scoring for logging multiple scores in crossvalidation.
- `wordvectors_train.py` script for training glove word vectors.
- `wordvectors_eval.py` script for evaluating trained glove vectors with the google analogy suite.
- `util.py` commonly used, general pythonic utility functions.
- `clean_output_dir.py`: removes empty dirs made as output when testing.

## Results
best score: 0.75F1.
