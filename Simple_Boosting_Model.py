'''
Created by Yutong Zhang on Nov 22.
The model is inspired by a MIT opencourse on AI, by professor Patrick Winston.
This Boosting model is mentioned on Lec 17 of that course.
Here is a link to that lecture:
https://youtu.be/UHBmv7qCey4

This is just simple modeling of this boosting idea, this won't even run.
It is an intermediate code between a sudo code and a runnable program.
Some of the implementation details are omitted.
The naming of variable is terrible. I am sorry for that.
'''
import numpy as np

class Boosting:
    # Construct a boosting object
    def __init__(self, classifers, test_data):
        self.classifers = classifers # all the classifers we can choose
        self.test_data = test_data
        self.error_weights = np.ones(test_data.shape(0))*(1/test_data.shape(0)) # the number of error_weights is the same as the number of test samples
        self.t = 1 # time index

    # A helper function that will UpdateErrorWeight
    def pickAClassifier():
        best_error_rate = 1 # initally, 100% error_rate, any classifer will be better than this
        for classifer in classifers:
            # run the test and generate new label_of_correctness based on that
            # if correct, label 1 on that sample index
            label_of_correctness = ... somthing new here

            # sum the error_weights, based on label_of_correctness, to get error_rate
            error_rate = sum error_weights over the wrong ones

            if error_rate >= best_error_rate:
                continue # skip this one
            else:
                picked = classifer
                picked_label_of_correctness = label_of_correctness
                best_error_rate = error_rate


        # rescale the error_weights, based on picked_label_of_correctness
        sum_of_correct_ones = ...
        sum_of_wrong_ones = ...
        for i in range(picked_label_of_correctness.shape(0)):
            # the sum of the rescaled error_weights over the correct ones is 1/2, and the sum of the rescaled error_weights over the wrong ones is 1/2 as well
            if picked_label_of_correctness(i) == 0:
                error_weights[i] /= sum_of_wrong_ones * 2
            else:
                error_weights[i] /= sum_of_wrong_ones * 2

        # remove the chose classifer on the list #
        remove picked from classifers
        t += 1 # increase the time index by 1, return the time index of the current calculation

        indices_of_the_wrong_ones = ... based on picked_label_of_correctness
        alpha = 0.5 * np.log((1 - best_error_rate) / best_error_rate)
        return picked, alpha, t-1


def getH(classifers, number_of_classifiers_in_H, test_data):
    classifers = a list of classifers
    # H is the combined classifer
    number_of_classifiers_in_H = number_of_classifiers_in_H
    test_data = test_data
    Boosting(classifers, test_data)

    H = any data structure that can hold alpha and ht
    for i in range(number_of_classifiers_in_H):
        H append Boosting.pickAClassifier()

    return H


def makePredictionBasedOnH(H, input):
    result = np.zeros(How many classes in this classfication problem) # each value in result is a probabilities

    for sample in input:
        make prediction on each ht in H
        result = alpha1*h1 + alpha2*h2 + alpha3*h3

    # I am not sure whether this model will fit in a multiple classfication problem, when the output is probabilities. That is to say, I am not sure whether this sum of probabilities is still one.

    return result;


main()
