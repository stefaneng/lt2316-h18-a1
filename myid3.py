# Module file for implementation of ID3 algorithm.
import pandas as pd
import numpy as np
from helper import *
from tree import Node
import pickle
# from sklearn.metrics import confusion_matrix, classification_report

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
            self.model = pickle.load(load_from)

    def _id3(self, X, attrs, target_attr, unique_values, depth = 1):
        """
        Takes in a dataframe (or subset) `X`
        that we want to split on one of `attrs`
        Target variable (class) we want to predict is in `target_attr`
        `unique_values` is a dictionary (attribute, list of unique values for attriute)
            Not sure if this is the best way to do it, but this will ensure that we add
            each value we see in the training set to every branch.
        """
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
        ig_max = -1
        max_attr = None
        for a in attrs:
            ig = info_gain_df(X, a, target_attr)
            if ig > ig_max:
                ig_max = ig
                max_attr = a
        root.child_attr = max_attr

        # Compute all the possible values for the attribute
        for u in unique_values[max_attr]:
            # Set each of the children to the results from
            # _id3 on
            # X[max_attr == u] as the data frame
            # Remove the current attribute from the array
            new_attrs = list(attrs.copy())
            new_attrs.remove(max_attr)
            examples = X[X[max_attr] == u]
            root.children[u] = Node()
            if len(examples) <= 0:
                # Add a leaf node with label = most common target value in the examples
                root.children[u].label = X[target_attr].value_counts().idxmax()

            else:
                root.children[u] = self._id3(examples, new_attrs, target_attr, unique_values, depth+1)

            # Set properties common to both cases
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
        # Compute the possible values for each attribute
        # Store in dictionary
        unique_values = {a: joined_df[a].unique() for a in joined_df}
        model = self._id3(joined_df, attrs, y.name, unique_values)
        self.model = model

    def _predict_one(self, instance):
        "Returns the class of a single given instance."
        current_node = self.model
        while not current_node.label:
            val = instance[current_node.child_attr]
            current_node = current_node.children[val]
        return current_node.label

    def predict(self, instance):
        # Returns the class of a given instance.
        # Raise a ValueError if the class is not trained.
        if not self.model:
            raise ValueError("Model is not trained.")
        preds = instance.apply(self._predict_one, axis=1)
        preds.name = "prediction"
        return preds

    def _confusion_matrix(self, predicted, actual):
        actual_classes = set(predicted)
        predicted_classes = set(actual)
        # Combine the possible classes in both y and predicted
        classes = actual_classes.union(predicted_classes)
        n = len(classes)
        conf_mat = pd.DataFrame(np.zeros((n, n)))
        conf_mat.columns = classes
        conf_mat.index = classes

        for (pred, actual) in zip(predicted, actual):
            conf_mat[pred][actual] += 1
        return conf_mat

    def _measures(self, confusion_matrix):
        true_pos = np.diag(confusion_matrix)
        false_pos = confusion_matrix.apply(lambda col: col.sum() - col[col.name], axis=0)
        false_neg = confusion_matrix.apply(lambda row: row.sum() - row[row.name], axis=1)
        count = confusion_matrix.values.sum()
        precision = true_pos / (true_pos + false_pos)
        # If we have no predicted values returns NaN (true positives + false positives = 0)
        #
        if sum(np.isnan(precision)) > 0:
            print("Precision and F-score are ill-defined and set to nan in labels with no predicted samples.")
        recall = true_pos / (true_pos + false_neg)
        if sum(np.isnan(recall)) > 0:
            print("Recall and F-score are ill-defined and being set to nan in labels with no predicted samples.")
        accuracy = true_pos.sum() / count
        f1 = 2 * precision * recall / (precision + recall)
        return {
        'accuracy': accuracy,
        'precision': precision.to_dict(),
        'recall': recall.to_dict(),
        'F1': f1.to_dict()
        }

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
        preds = self.predict(X)
        result['confusion-matrix'] = self._confusion_matrix(preds, y)
        # Add accuracy, recall and precision and update the result
        result.update(self._measures(result['confusion-matrix']))
        if display:
            print(result)
            # TODO: Remove this
            #print("sklearn implementation:", classification_report(y, preds, target_names=result['confusion-matrix'].columns.values))
            #print("sklearn implementation:", confusion_matrix(y, preds))

        return result

    def __str__(self):
        # Returns a readable string representation of the trained
        # decision tree or "ID3 untrained" if the model is not trained.
        return str(self.model)

    def save(self, output):
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        pickle.dump(self.model, output)
