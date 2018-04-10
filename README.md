# SENTiVENT: Company-specific event detection in economic news
Economic event type detection on the SentiFM dataset using neural and other ML methods.

This repo includes data and code for company-specific sentence level event detection for the [English SentiFM dataset](https://osf.io/enu2k/).

It runs complete experiments with word-vector creation & evaluation, hyperparameter optimization in crossvalidation and tests on holdout.

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
    TADAH, we now have a properly optimised deep learning stack!
    The Python executable path with the deps can be found at `.//yes/bin/python`.

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