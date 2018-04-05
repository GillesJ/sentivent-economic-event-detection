# Performance log
Bookkeeping for architectures and performance.

## glove.6B: 3-fold random search 32 iterations
- {'optimizer': (<class 'keras.optimizers.Adam'>, {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'decay': 0.0}), 'lstm_units': 536, 'lstm_recurrent_dropout': 0.0, 'lstm_dropout': 0.0, 'epochs': 32, 'bidirectional': True, 'batch_size': 64}

```
/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-18_22-50-14
CV:
AVG accuracy        f1  nfoldstrained  precision    recall       auc
    0.854747  0.625002  3              0.706319   0.571484  0.782764
    
precision_per_label
[0.845225274687, 0.592592592593, 0.767857142857, 0.43552582683, 0.767238860527, 0.663695062, 0.765322814464, 0.671717171717, 0.797103870788, 0.756913843481]

recall_per_label
[0.821598973405, 0.394033211371, 0.60202414516, 0.307574238627, 0.657634632722, 0.492652802557, 0.641173740904, 0.436157796452, 0.709226190476, 0.652767906499]

f1_per_label
[0.828292816003, 0.467249596282, 0.674815579253, 0.353406807581, 0.704942861676, 0.56346816405, 0.694149484247, 0.523218390805, 0.741161567126, 0.699317988212]

HOLDOUT
                   precision    recall  f1-score   support

        BuyRating       0.86      0.82      0.84        22
             Debt       0.00      0.00      0.00         2
         Dividend       0.50      0.55      0.52        11
MergerAcquisition       0.40      0.32      0.36        25
           Profit       0.82      0.79      0.81        58
 QuarterlyResults       0.77      0.68      0.72        34
      SalesVolume       0.84      0.73      0.78        51
  ShareRepurchase       1.00      0.67      0.80         6
      TargetPrice       0.75      0.75      0.75         4
         Turnover       0.90      0.73      0.81        26

      avg / total       0.77      0.69      0.72       239
```

## sentifm.glove.en_maintype_holdin.200d random search 32 iterations
- Best CV f1-score: 0.571913033078
```
CV:
    accuracy        f1  nfoldstrained  precision    recall       auc
    0.848261  0.571913  3              0.714949   0.486450  0.740772

f1_per_label  \
[0.761781253268, 0.406904761905, 0.597395369198, 0.256500635947, 0.673880448005, 0.499842238552, 0.707478404373, 0.475219533091, 0.685301999588, 0.654825686857]

precision_per_label  \
[0.862343290512, 0.639814814815, 0.754316816817, 0.393706535011, 0.780436339075, 0.667030135457, 0.803796255344, 0.640572390572, 0.87077668596, 0.736694860117]

recall_per_label  \
[0.697552793854, 0.298339431466, 0.499248174655, 0.193411559201, 0.597048873891, 0.403325912068, 0.634133401788, 0.382819794585, 0.566071428571, 0.592553107478]
    
HOLDOUT
                   precision    recall  f1-score   support

        BuyRating       0.91      0.91      0.91        22
             Debt       1.00      0.50      0.67         2
         Dividend       0.50      0.36      0.42        11
MergerAcquisition       0.32      0.24      0.27        25
           Profit       0.75      0.81      0.78        58
 QuarterlyResults       0.87      0.38      0.53        34
      SalesVolume       0.92      0.67      0.77        51
  ShareRepurchase       0.80      0.67      0.73         6
      TargetPrice       1.00      0.50      0.67         4
         Turnover       0.95      0.69      0.80        26

      avg / total       0.79      0.62      0.68       239
```

## No pretrained IN PAPER
- CV BEST ESTIMATOR: <sentifmdetect.classifier.KerasClassifierCustom object at 0x7fafd41287f0>

  CV BEST PARAMETERS: {'wvec': 200, 'optimizer': (<class 'keras.optimizers.Adam'>, {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'decay': 0.0}), 'lstm_units': 268, 'lstm_recurrent_dropout': 0.2, 'lstm_dropout': 0
.0, 'epochs': 128, 'bidirectional': True, 'batch_size': 64}

```
/home/gilles/repos/sentifmdetect17/sentifmdetect/output/en_maintype_2017-12-22_16-59-07"
CV:     
    accuracy        f1  nfoldstrained  precision    recall       auc
    0.829476  0.527131  3              0.619543   0.469478  0.731070

f1_per_label  \
[0.700838613882, 0.304542304542, 0.609317621582, 0.227618734188, 0.631494803763, 0.529719946045, 0.653743074103, 0.319667260844, 0.688817330211, 0.605550080486]

precision_per_label  \
[0.849206349206, 0.402396514161, 0.667224080268, 0.291714669739, 0.688085676037, 0.622157164869, 0.687230769231, 0.515476190476, 0.761936339523, 0.709999636773]

recall_per_label  \
[0.597130356897, 0.263298620884, 0.563836477987, 0.193066757804, 0.584319661216, 0.465766431744, 0.633122702395, 0.233193277311, 0.631349206349, 0.529692805812]


HOLDOUT:
                   precision    recall  f1-score   support

        BuyRating       0.81      0.59      0.68        22
             Debt       0.33      0.50      0.40         2
         Dividend       0.75      0.55      0.63        11
MergerAcquisition       0.21      0.12      0.15        25
           Profit       0.83      0.33      0.47        58
 QuarterlyResults       0.67      0.35      0.46        34
      SalesVolume       0.86      0.61      0.71        51
  ShareRepurchase       0.60      0.50      0.55         6
      TargetPrice       1.00      0.50      0.67         4
         Turnover       0.88      0.58      0.70        26

      avg / total       0.74      0.44      0.54       239
```

## sentifm.glove.en_maintype_holdin.200d with above parameters: 2fold
- {'optimizer': (<class 'keras.optimizers.Adam'>, {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'decay': 0.0}), 'lstm_units': 268, 'lstm_recurrent_dropout': 0.0, 'lstm_dropout': 0.0, 'epochs': 32, 'bidirectional': True, 'batch_size': 64}

CV score: 0.491329605884
```
                   precision    recall  f1-score   support

        BuyRating       0.93      0.64      0.76        22
             Debt       0.50      0.50      0.50         2
         Dividend       0.40      0.36      0.38        11
MergerAcquisition       0.40      0.08      0.13        25
           Profit       0.74      0.72      0.73        58
 QuarterlyResults       0.77      0.68      0.72        34
      SalesVolume       0.84      0.80      0.82        51
  ShareRepurchase       0.67      0.67      0.67         6
      TargetPrice       0.67      0.50      0.57         4
         Turnover       0.77      0.77      0.77        26

      avg / total       0.73      0.64      0.67       239
```

## NAACL paper 3fold random search prelim
- CV Results:        {'f1': 0.65042106744474659, 'precision': 0.68054018086553592, 'recall': 0.63356456458241195, 'accuracy': 0.86145588728614564, 'auc': 0.81329140823490764}
Pipeline:       {'optimizer': (<class 'keras.optimizers.Adam'>, {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'decay': 0.0}), 'lstm_units': 536, 'lstm_recurrent_dropout': 0.0, 'lstm_dropout': 0.0, 'epochs': 32, 'bidirectional': True, 'batch_size': 64, 'build_fn': <function create_pretrained_emb_lstm at 0x7f5b3d33ca60>}

- CV Results:        {'f1': 0.64592323847758326, 'precision': 0.72623476465915204, 'recall': 0.59114599246760036, 'accuracy': 0.86782958738678295, 'auc': 0.79284796266318791}
Pipeline:       {'optimizer': (<class 'keras.optimizers.Adam'>, {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'decay': 0.0}), 'lstm_units': 536, 'lstm_recurrent_dropout': 0.0, 'lstm_dropout': 0.2, 'epochs': 128, 'bidirectional': False, 'batch_size': 512, 'build_fn': <function create_pretrained_emb_lstm at 0x7f5b3d33ca60>}

- CV Results:        {'f1': **0.65711030569733242**, 'precision': 0.68758884367609796, 'recall': 0.6451331282299122, 'accuracy': 0.85306943978530692, 'auc': 0.8184430501074601}
Pipeline:       {'optimizer': (<class 'keras.optimizers.Adam'>, {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'decay': 0.0}), 'lstm_units': 536, 'lstm_recurrent_dropout': 0.2, 'lstm_dropout': 0.2, 'epochs': 32, 'bidirectional': False, 'batch_size': 256, 'build_fn': <function create_pretrained_emb_lstm at 0x7fde0535b9d8>}

- CV Results:        {'f1': 0.63125612669087849, 'precision': 0.71271841759624466, 'recall': 0.57240306793207663, 'accuracy': 0.8456893659845689, 'auc': 0.78278024843073912}
Pipeline:       {'optimizer': (<class 'keras.optimizers.Adam'>, {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'decay': 0.0}), 'lstm_units': 536, 'lstm_recurrent_dropout': 0.2, 'lstm_dropout': 0.0, 'epochs': 128, 'bidirectional': False, 'batch_size': 64, 'build_fn': <function create_pretrained_emb_lstm at 0x7f5b3d33ca60>}

## Manual setting of params (no hyperparameter search)
1. Bidirectional LSTM with glove.6B.100d pretrained word embeddings:
```
model = Sequential()
model.add(
    Embedding(len(word_index)+1, settings.EMB_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH))  # irony
model.add(Bidirectional(LSTM(int(MAX_SEQUENCE_LENGTH * 2), dropout=0.2, recurrent_dropout=0.2)))
# model.add(Bidirectional(LSTM(int(MAX_SEQUENCE_LENGTH*2))))
model.add(Dense(len(labelenc.classes_), activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
logging.info(model.summary())
model.fit(x_train, y_train, epochs=20, batch_size=64, class_weight="balanced", verbose=1)
```
training:
loss: 0.0089 acc: 0.9970

Randomized holdout test:
```
                precision  recall   f1-score   support

Buyrating0          0.92      0.75      0.83        16
Debt1               0.67      0.33      0.44         6
Dividend2           0.63      0.75      0.69        16
MergerAcquisition3  0.46      0.55      0.50        22
No_event4           0.94      0.92      0.93       848
Profit5             0.76      0.76      0.76        79
QuarterlyResults6   0.61      0.77      0.68        30
SalesVolume7        0.78      0.66      0.71        53
ShareRepurchase8    0.60      0.75      0.67         4
TargetPrice9        1.00      0.67      0.80         6
TurnOver10          0.71      0.60      0.65        25

avg / total         0.89      0.87      0.88      1105
```
2. idem 1, with 25 instead of 20 epochs (worse):
```
        BuyRating       1.00      0.69      0.81        16
             Debt       0.67      0.33      0.44         6
         Dividend       0.63      0.75      0.69        16
MergerAcquisition       0.48      0.55      0.51        22
           NoType       0.93      0.92      0.92       848
           Profit       0.81      0.73      0.77        79
 QuarterlyResults       0.61      0.67      0.63        30
      SalesVolume       0.80      0.66      0.72        53
  ShareRepurchase       0.75      0.75      0.75         4
      TargetPrice       1.00      0.67      0.80         6
         Turnover       0.71      0.60      0.65        25

      avg / total       0.89      0.86      0.87      1105
```

3. idem 1, with lstm_units=MAX_SEQUENCE, BEST
```
                   precision    recall  f1-score   support

        BuyRating       0.86      0.75      0.80        16
             Debt       1.00      0.33      0.50         6
         Dividend       0.67      0.75      0.71        16
MergerAcquisition       0.60      0.55      0.57        22
           NoType       0.94      0.92      0.93       848
           Profit       0.78      0.78      0.78        79
 QuarterlyResults       0.58      0.70      0.64        30
      SalesVolume       0.77      0.70      0.73        53
  ShareRepurchase       0.60      0.75      0.67         4
      TargetPrice       0.75      0.50      0.60         6
         Turnover       0.74      0.68      0.71        25

      avg / total       0.89      0.87      0.88      1105
avg / total w/o NoType  0.74      0.70      0.71       257
```

