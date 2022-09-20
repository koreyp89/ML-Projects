import numpy as np
import pandas as pd
import sys


def make_test_set():
    training_data = pd.read_csv(sys.argv[1], sep="\t", header=[0])
    print(training_data.head())
    global COLUMN_HEADERS
    COLUMN_HEADERS = training_data.columns
    return training_data



if __name__ == "__main__":
    training_dat = make_test_set()
    training_data = training_dat.values

    class_column = training_data[:, -1]
    unique_val, counts = np.unique(class_column, return_counts=True)

    total = counts[0] + counts[1]
    if unique_val[0] == 0:
        class_0 = counts[0]
        class_prob_0 = counts[0] / total
    else:
        class_0 = counts[1]
        class_prob_0 = counts[1] / total

    class_1 = total - class_0
    class_prob_1 = 1 - class_prob_0
    rows, columns = training_data.shape
    conditionals = {}

    for header in COLUMN_HEADERS:
        conditionals[header] = {}
        conditionals[header][0] = {}
        conditionals[header][1] = {}
    for j in range(columns):
        attribute_and_class0 = 0
        attribute_and_class1 = 0
        for i in range(rows):
            if training_data[i][j] == 1:
                if class_column[i] == 1:
                    attribute_and_class1 += 1
                else:
                    attribute_and_class0 += 1
        conditionals[COLUMN_HEADERS[j]][1][1] = attribute_and_class1 / class_1
        conditionals[COLUMN_HEADERS[j]][1][0] = attribute_and_class0 / class_0
        conditionals[COLUMN_HEADERS[j]][0][1] = 1 - conditionals[COLUMN_HEADERS[j]][1][1]
        conditionals[COLUMN_HEADERS[j]][0][0] = 1 - conditionals[COLUMN_HEADERS[j]][1][0]
    print("P(class=0)=" + str(round(class_prob_0, 2)), end=' ')
    for header in COLUMN_HEADERS[:len(COLUMN_HEADERS) - 1]:
        for i in range(2):
            print("P(" + header + "=" + str(i) + "|0)=" + str.format("{:.2f}", conditionals[header][i][0]), end=' ')
    print()
    print("P(class=1)=" + str(round(class_prob_1, 2)), end=' ')
    for header in COLUMN_HEADERS[:len(COLUMN_HEADERS) - 1]:
        for i in range(2):
            print("P(" + header + "=" + str(i) + "|1)=" + str.format("{:.2f}", conditionals[header][i][1]), end=' ')
    attributes = COLUMN_HEADERS[:len(COLUMN_HEADERS) - 1]

