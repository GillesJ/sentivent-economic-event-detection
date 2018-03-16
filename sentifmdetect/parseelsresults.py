# [precision, recall]
import numpy as np

svm_linearkernel = {
"BuyRating": [0.952380952381, 0.909090909091,],
"Debt": [0.50, 1.00,],
"Dividend": [0.615384615385, 0.727272727273,],
"MergerAcquisition": [0.555555555556, 0.40,],
"Profit": [0.754385964912, 0.741379310345,],
"QuarterlyResults": [0.818181818182, 0.529411764706,],
"SalesVolume": [0.883720930233, 0.745098039216,],
"ShareRepurchase": [1.00, 0.50,],
"TargetPrice": [1.00, 0.75,],
"Turnover": [0.909090909091, 0.769230769231,],
}

svm_optimisedrbf = {
"BuyRating": [0.952380952381, 0.909090909091,],
"Debt": [0.50, 1.00,],
"Dividend": [0.538461538462, 0.636363636364,],
"MergerAcquisition": [0.00, 0.00,],
"Profit": [0.80, 0.758620689655,],
"QuarterlyResults": [0.826086956522, 0.558823529412,],
"SalesVolume": [0.942857142857, 0.647058823529,],
"ShareRepurchase": [1.00, 0.50,],
"TargetPrice": [1.00, 0.75,],
"Turnover": [0.869565217391, 0.769230769231,],
}
# ^\s*([a-zA-Z]+)\s+(\d\.\d\d)\s+(\d\.\d\d)\s+\d\.\d\d\s+\d+ | "$1": [$2, $3],
lstm_6b = {
    "BuyRating": [0.86, 0.82],
    "Debt": [0.00, 0.00],
    "Dividend": [0.50, 0.55],
    "MergerAcquisition": [0.40, 0.32],
    "Profit": [0.82, 0.79],
    "QuarterlyResults": [0.77, 0.68],
    "SalesVolume": [0.84, 0.73],
    "ShareRepurchase": [1.00, 0.67],
    "TargetPrice": [0.75, 0.75],
    "Turnover": [0.90, 0.73],
}

lstm_sentifm = {
    "BuyRating": [0.91, 0.91],
    "Debt": [1.00, 0.50],
    "Dividend": [0.50, 0.36],
    "MergerAcquisition": [0.32, 0.24],
    "Profit": [0.75, 0.81],
    "QuarterlyResults": [0.87, 0.38],
    "SalesVolume": [0.92, 0.67],
    "ShareRepurchase": [0.80, 0.67],
    "TargetPrice": [1.00, 0.50],
    "Turnover": [0.95, 0.69],
}

lstm_no = {
    "BuyRating": [0.81, 0.59],
    "Debt": [0.33, 0.50],
    "Dividend": [0.75, 0.55],
    "MergerAcquisition": [0.21, 0.12],
    "Profit": [0.83, 0.33],
    "QuarterlyResults": [0.67, 0.35],
    "SalesVolume": [0.86, 0.61],
    "ShareRepurchase": [0.60, 0.50],
    "TargetPrice": [1.00, 0.50],
    "Turnover": [0.88, 0.58],
}

f1_all = []
p_all = []
r_all = []
for k, v in lstm_sentifm.items():
    if v[0] == 0 and v[1] == 0:
        f1 = 0.0
    else:
        f1 = 2*(v[0]*v[1])/(v[0]+v[1])
    f1_all.append(f1)
    p_all.append(v[0])
    r_all.append(v[1])
print("precision avg: {}".format(np.mean(p_all)))
print("recall avg: {}".format(np.mean(r_all)))
print("f1 avg: {}".format(np.mean(f1_all)))