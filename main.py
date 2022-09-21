import numpy as np
import pandas as pd
import sys
import math


def make_test_set(file_location):
    training_data = pd.read_csv(file_location, sep="\t", header=[0])
    return training_data


class Bayes_reasoner:

    def __init__(self, training_file_location):
        self.matrix = make_test_set(training_file_location)
        self.training_set = self.matrix.values
        self.attributes = self.matrix.columns[:len(self.matrix.columns)-1]
        self.class_priors = {}
        self.initialize_class_priors()
        self.get_class_priors()
        self.conditional_probabilities = {}
        self.initialize_conditionals()
        self.get_conditionals()
        self.print_conditionals()

    def initialize_conditionals(self):
        for attribute in self.attributes:
            self.conditional_probabilities[attribute] = {}
            for j in range(2):
                self.conditional_probabilities[attribute][j] ={}
                for i in range(2):
                    self.conditional_probabilities[attribute][j][i] = 0

    def argmax_classes(self, example):
        argmax = 0
        argmax_value = 0
        for x in range(2):
            summation = math.log(self.class_priors[x], 2)
            for i in range(self.attributes):
                summation += math.log(self.conditional_probabilities[self.attributes[i]][example[i]][x])
            if summation > argmax_value:
                argmax_value = summation
                argmax = x
        return argmax

    def get_conditionals(self):
        denom = len(self.training_set[:,-1])
        p_of_0 = self.class_priors[0]
        p_of_1 = self.class_priors[1]
        for i in range(len(self.attributes)):
            attribute0_and_class0 = 0
            attribute1_and_class0 = 0
            attribute1_and_class1 = 0
            attribute0_and_class1 = 0
            for j in range(len(self.training_set[:,-1])):
                if self.training_set[j][i] == 0:
                    if self.training_set[j][-1] == 0:
                        attribute0_and_class0 += 1
                    else:
                        attribute0_and_class1 += 1
                else:
                    if self.training_set[j][-1] == 0:
                        attribute1_and_class0 += 1
                    else:
                        attribute1_and_class1 += 1
            self.conditional_probabilities[self.attributes[i]][0][0] = (attribute0_and_class0/(denom*p_of_0))
            self.conditional_probabilities[self.attributes[i]][0][1] = (attribute0_and_class1/(denom*p_of_1))
            self.conditional_probabilities[self.attributes[i]][1][0] = (attribute1_and_class0/(denom*p_of_0))
            self.conditional_probabilities[self.attributes[i]][1][1] = (attribute1_and_class1/(denom*p_of_1))

    def get_class_priors(self):
        zero_total = 0
        for i in self.training_set[: , -1]:
            zero_total += 1 if i == 0 else 0
        self.class_priors[0] = zero_total/len(self.training_set[: , -1])
        self.class_priors[1] = 1 - self.class_priors[0]

    def initialize_class_priors(self):
        self.class_priors[0] = 0
        self.class_priors[1] = 0

    def print_conditionals(self):
        print("P(class=0)=" + str(round(self.class_priors[0], 2)), end=' ')
        for att in self.attributes:
            for i in range(2):
                print("P(" + att + "=" + str(i) + "|0)=" + str.format("{:.2f}", self.conditional_probabilities[att][i][0]), end=' ')
        print()
        print("P(class=1)=" + str(round(self.class_priors[1], 2)), end=' ')
        for att in self.attributes:
            for i in range(2):
                print("P(" + att + "=" + str(i) + "|1)=" + str.format("{:.2f}", self.conditional_probabilities[att][i][1]), end=' ')


if __name__ == "__main__":
    Bayes_reasoner(sys.argv[1])
