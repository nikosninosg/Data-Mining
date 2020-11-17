#!/usr/bin/python3
# -*- coding: utf-8 -*

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from matplotlib import style


with open('winequality-red.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(row)


def Build_Data_Set(features = ["DE Ratio", "Trailing P/E"]):
    data_df = pd.DataFrame.from_csv("winequality-red.csv")

    data_df = data_df[:100]

    X = np.array(data_df[features].values)

    y = (data_df["Status"]
         .replace("underperform",0)
         .replace("outperform",1)
         .values.tolist())


    return X,y


def Analysis():
    X, y = Build_Data_Set()

    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(X, y)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
    yy = a * xx - clf.intercept_[0] / w[1]

    h0 = plt.plot(xx, yy, "k-", label="non weighted")

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.ylabel("Trailing P/E")
    plt.xlabel("DE Ratio")
    plt.legend()

    plt.show()


Analysis()

"""
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}');
            line_count += 1;
        else:
            print(f"{row[0]} \n ");
            #print(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} {row[6]} {row[7]} {row[8]} {row[9]} {row[10]} {row[11]} ");
            line_count += 1;

print(f'Processed {line_count} lines.');
"""

