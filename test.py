# Testing file for decision tree induction.
# This file is intended to help you figure out what is expected
# for you to pass the assignment. I don't recommend you change it, but
# rather copy it and make your own tests or write tests afresh.
# We can/will test your code using other data.
# If you see a bug or something unclear here, contact Asad via GUL or email.

import os, sys
import pandas as pd
import numpy as np
from myid3 import DecisionTree # What you will write in the myid3.py file.

# Loading the data.
balancedata = pd.read_csv("balance-scale.data", index=None, columns=None)

y = balancedata[0]
X = balancedata[1:4]


