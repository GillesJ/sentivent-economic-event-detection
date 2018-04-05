# Binary experiment results

Exp.1: 
- OPT: "/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-28_11:03:30_CEST"
- PARAM GRID + SEARCHCV
    ```python
     param_grid = {
            "wvec": [settings.EMB_FP["glove.6B.300d"], settings.EMB_FP["glove.en_maintype_w15_lr0.25_ep20.200d.glovemodel"]],
            "bidirectional": [True, False],
            "lstm_units": [int(settings.MAX_SEQUENCE_LENGTH) * 4, int(settings.MAX_SEQUENCE_LENGTH * 8)],
            "lstm_dropout": [0.0, 0.2],
            "lstm_recurrent_dropout": [0.0, 0.2],
            "optimizer": [
                (Adam, {"lr":0.001, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":0.0}), # we do not init any objects in params so that they are visible as dict to the CV.get_params().
                # (Adam, {"lr":0.002, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":1.0}), # this does not seem to do better, ever
            ],
            "batch_size": [32, 64],
            "epochs": [32, 64],
        }
    searchcv = classifier.KerasRandomizedSearchCV(
            clf,
            param_distributions=param_grid,
            n_iter=16,
            cv=settings.KFOLD,
            scoring=scorer.my_scorer,
            fit_params=dict(callbacks=[settings.EARLY_STOP, ]), # hack for adding keras callbacks
            verbose=0,
            error_score=0, # value 0 ignores failed fits and move on to next fold
            return_train_score=False, # circumvents bug in sklearn, prevent return the meaningless train score which will always be near perfect
            random_state=settings.RANDOM_SEED,
            n_jobs=1,
            )
    ```
- LABEL BuyRating:
    - BEST SETTINGS `INFO:root:CV BEST PARAMETERS: {'wvec': '/home/gilles/corpora/word_embeddings/glove_stanford/glove.6B/glove.6B.300d.txt', 'optimizer': (<class 'keras.optimizers.Adam'>, {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'decay': 0.0}), 'lstm_units': 536, 'lstm_recurrent_dropout': 0.2, 'lstm_dropout': 0.0, 'epochs': 64, 'bidirectional': False, 'batch_size': 64}`
    - CV BEST SCORE: 0.8491175306755345
    - HOLDOUT SCORE: `{'f1': 0.8095238095238095, 'precision': 0.85, 'recall': 0.7727272727272727, 'accuracy': 0.9919517102615694, 'roc_auc': 0.8848204264870931}`

Exp. 2:
- REST RUN LIMITED ITERATIONS AND PARAMS
- OPT: /home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-29_18:25:07_CEST/metadata.json
- PARAM GRID + SEARCHCV
```python
    param_grid = {
        "wvec": [settings.EMB_FP["glove.6B.300d"], settings.EMB_FP["glove.en_maintype_w15_lr0.25_ep20.200d.glovemodel"]],
        "bidirectional": [True, False],
        "lstm_units": [int(settings.MAX_SEQUENCE_LENGTH) * 4, int(settings.MAX_SEQUENCE_LENGTH * 8)],
        "lstm_dropout": [0.0, 0.2],
        "lstm_recurrent_dropout": [0.0, 0.2],
        "optimizer": [
            (Adam, {"lr":0.001, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":0.0}), # we do not init any objects in params so that they are visible as dict to the CV.get_params().
            # (Adam, {"lr":0.002, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":1.0}), # this does not seem to do better, ever
        ],
        "batch_size": [32, 64],
        "epochs": [32, 64],
    }

    searchcv = classifier.KerasRandomizedSearchCV(
        clf,
        param_distributions=param_grid,
        n_iter=8,
        cv=settings.KFOLD,
        scoring=scorer.my_scorer,
        fit_params=dict(callbacks=[settings.EARLY_STOP, ]), # hack for adding keras callbacks
        verbose=0,
        error_score=0, # value 0 ignores failed fits and move on to next fold
        return_train_score=False, # circumvents bug in sklearn, prevent return the meaningless train score which will always be near perfect
        random_state=settings.RANDOM_SEED,
        n_jobs=1,
        )
```
3. RUN PLUS EARLY STOP MORE
- RUN OPT: "/home/gilles/repos/sentifmdetect17/output/en_maintype_2018-03-31_00:54:36_CEST/metadata.json"
- TRAIN DUR: +23h
- PARAMS AND CV:
```python
param_grid = {
    "wvec": [settings.EMB_FP["glove.6B.300d"]],
    "bidirectional": [True, False],
    "lstm_units": [int(settings.MAX_SEQUENCE_LENGTH) * 4, int(settings.MAX_SEQUENCE_LENGTH * 8)],
    "lstm_dropout": [0.0, 0.2],
    "lstm_recurrent_dropout": [0.0, 0.2],
    "optimizer": [
        (Adam, {"lr":0.001, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":0.0}), # we do not init any objects in params so that they are visible as dict to the CV.get_params().
        # (Adam, {"lr":0.002, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-08, "decay":1.0}), # this does not seem to do better, ever
    ],
    "batch_size": [128, 256],
    "epochs": [32],
}
    
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
```

4. EVEN MORE SETTINGS
- OPT: /home/gilles/repos/sentifmdetect17/output/en_maintype_2018-04-02_13:50:21_CEST/metadata.json
- DURATION: failed OOM
- PARAMS + CV
```python
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
searchcv = classifier.KerasRandomizedSearchCV(
    clf,
    param_distributions=param_grid,
    n_iter=64,
    cv=settings.KFOLD,
    scoring=scorer.my_scorer,
    verbose=0,
    error_score=0,  # value 0 ignores failed fits and move on to next fold
    return_train_score=False, # circumvent bug sklearn, prevent return of near-perfect meaningless train score
    random_state=settings.RANDOM_SEED,
    n_jobs=1,
)
```

5. EVEN MORE SETTINGS RETRY WITH 32 CV ITERATIONS
- OPT: /home/gilles/repos/sentifmdetect17/output/en_maintype_2018-04-02_23:00:22_CEST/metadata.json
- DURATION: failed OOM
- PARAMS + CV
```python
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
```