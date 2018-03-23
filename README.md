# SENTiVENT: SentiFM economic event detection '17
Economic event type detection on the SentiFM dataset using neural and other ML methods.

Experiments with validation set and crossvalidation.

We use TensorFlow, not Theano, with fp16(test this). Even though Theano benchmarks much better on RNNs according to some, 
Tensorflow install actually works (Theano caused massive unresolved issues).

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
    The Python executable path with the deps can be found at `/home/gilles/repos/sentifmdetect17/yes/bin/python`.

## Contents
- `settings.py` for defining the experimental constants for crossvalidation optim. & testing, & wordvector training.
- `featurize.py` feature engineering: tokenisation, indexing, sequencing & making the embedding matrix.
- `crossvalidate.py` run validation test & crossvalidation experiment using featurized data.
- `datahandler.py` loading, parsing, writing, making splits and general handling of dataset.
- `classifier.py` custom sklearn-compatible classifiers and classifier handling.
- `scorer.py` custom classifier scoring for logging multiple scores in crossvalidation.
- `wordvectors_train.py` script for training glove word vectors.
- `wordvectors_eval.py` script for evaluating trained glove vectors with the google analogy suite.
- `util.py` commonly used, general pythonic utility functions.
- `clean_output_dir.py`: removes empty dirs made as output when testing.

## Usage
cf. above
[] finish this
[] add what needs to be done for a crossvalidate.py run
[] specifically add which settings and resources are ****set manually

## Results
see NAACL18 paper submission

DEPRECATED: Theano install:
- `./sentifmdetect17` We will install all modules and non-system dependencies through conda in this folder.
- Conda: install miniconda (latest 3.6.3) from site.
    - `cd ~ && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    - `sh Miniconda3-latest-Linux-x86_64.sh`
    - Do not forget to add to .zshrc:
        ```bash
        # added by Miniconda3 installer
        export PATH="/home/gilles/miniconda3/bin:$PATH"
        export PATH="/home/gilles/repos/sentifmdetect17/yes/bin:$PATH"
        ```
- create virtualenv `conda create -n sentifm17 python=3.6`
- active `source activate sentifm17`
- `conda install numpy scipy mkl mkl-service nose libgpuarray nltk gensim pandas`
- `pip install pycuda parameterized fuzzywuzzy python-levenshtein sklearn-deap; pip install git+https://github.com/lebedov/scikit-cuda.git#egg=scikit-cuda git+https://github.com/maciejkula/glove-python.git` 
    scikit-cuda is optional but version 0.5.2. (currently dev) is required for Theano.
- `conda install theano pygpu keras`
- make a file at `~/.theanorc` containing:
    ```
    [global]
    device = cuda
    floatX = float32
    
    [cuda] 
    root = /usr/local/cuda-9.1/bin
    
    [nvcc]
    fastmath = True
    ```
- `export PYTHONPATH=: && EXPORT MKL_THREADING_LAYER=GNU`
- test Theano with `python -c "import theano; theano.test()"`
- TROUBLESHOOT: Make sure you are using the python binary of the virtualenv `python -c "import sys; print(sys.executable)"`
- TROUBLESHOOT: If CUDNN cannot by found by Theano: The easiest is to include them in your CUDA installation. Copy the `*.h` files to `CUDA_ROOT/include` and the `*.so*` files to `CUDA_ROOT/lib64` (by default, CUDA_ROOT is `/usr/local/cuda` on Linux). `sudo ln -s /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64`
- TROUBLESHOOT: run `theano-cache purge` on changing environment vars or theano flags, or when theano has a segmentation fault.

