# Module file for implementation of ID3 algorithm.
import pandas as pd
import numpy as np
from helper import *
from tree import Node

# You can add optional keyword parameters to anything, but the original
# interface must work with the original test file.
# You will of course remove the "pass".

import os, sys
import numpy
# You can add any other imports you need.

class DecisionTree:
    def __init__(self, load_from=None):
        # Fill in any initialization information you might need.
        #
        # If load_from isn't None, then it should be a file *object*,
        # not necessarily a name. (For example, it has been created with
        # open().)
        self.model = None
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")

    def _id3(self, X, attrs, target_attr, depth = 1):
        # All positive or all negative examples
        root = Node()
        root.depth = depth
        if len(X.groupby(target_attr)) == 1:
            # Return single-node tree Root with label = first values in X[target_attr]
            if len(X[target_attr]) > 0:
                root.label = X[target_attr].iloc[0]
            return root
        if len(attrs) <= 0:
            # label = most common value of the target attribute in the examples.
            label = X[target_attr].value_counts().idxmax()
            return root

        # Compute the maximum information gain attribute
        ig_max = 0
        max_attr = None
        for a in attrs:
            ig = info_gain_df(X, a, target_attr)
            if ig > ig_max:
                ig_max = ig
                max_attr = a
        root.child_attr = max_attr

        # Compute all the possible values for the attribute
        for u in X[max_attr].unique():

            # Set each of the children to the results from
            # _id3 on
            # X[max_attr == u] as the data frame
            # Remove the current attribute from the array
            new_attrs = list(attrs.copy())
            new_attrs.remove(max_attr)

            root.children[u] = self._id3(X[X[max_attr] == u], new_attrs, target_attr, depth+1)
            root.children[u].value = u
            root.children[u].attr = max_attr
        return root

    def train(self, X, y, attrs, prune=False):
        # Doesn't return anything but rather trains a model via ID3
        # and stores the model result in the instance.
        # X is the training data, y are the corresponding classes the
        # same way "fit" worked on SVC classifier in scikit-learn.
        # attrs represents the attribute names in columns order in X.
        #
        # Implementing pruning is a bonus question, to be tested by
        # setting prune=True.
        #
        # Another bonus question is continuously-valued data. If you try this
        # you will need to modify predict and test.
        joined_df = pd.concat([X, y], axis=1)
        model = self._id3(joined_df, attrs, y.name)
        self.model = model

    def predict(self, instance):
        # Returns the class of a given instance.
        # Raise a ValueError if the class is not trained.
        pass

    def test(self, X, y, display=False):
        # Returns a dictionary containing test statistics:
        # accuracy, recall, precision, F1-measure, and a confusion matrix.
        # If display=True, print the information to the console.
        # Raise a ValueError if the class is not trained.
        result = {'precision':None,
                  'recall':None,
                  'accuracy':None,
                  'F1':None,
                  'confusion-matrix':None}
        if display:
            print(result)
        return result

    def __str__(self):
        # Returns a readable string representation of the trained
        # decision tree or "ID3 untrained" if the model is not trained.
        return str(self.model)

    def save(self, output):
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        pass
