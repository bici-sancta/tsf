
import pandas as pd
import numpy as np
from metrics import Metrics

# ... naïve model -- prediction is constant value forward from last value of train period

class Primitive(Metrics):

    def __init__(self) :
        self.value_ = None

    def fit(self, X, omega = None):
        """
        Fit model coefficients.

        Arguments:
        X: 2D pandas dataframe - dates X[0] and values X[1]
        omega : last value (date) in training period
        """

# ...   self.data and .target are required to be defined for use in Metrics()

        self.data = np.array(X[X.columns[1]])
        self.target = np.array(X[X.columns[1]])

# ...   sort in ascending date order, drop all rows after omega, use last value remaining

        try :
            X_sorted = X.sort_values(X.columns[0])
            X_red = X_sorted[X_sorted[X_sorted.columns[0]] <= omega]
            row_num = len(X_red)
            last_value = X_red.iloc[row_num - 1][X_red.columns[1]]
            self.value_ = last_value
        except Exception as err :
            print (err)

    def predict(self, X):
        """Output model prediction.

        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return [self.value_] * len(X)

# ... mean historic value model -- prediction is constant value forward from mean value of train period

class Mean(Metrics):

    def __init__(self) :
        self.value_ = None

    def fit(self, X):
        """
        Fit model coefficients.

        Arguments:
        X: 2D pandas dataframe - dates X[0] and values X[1]
        mean_value : mean value of entire training period
        """

# ...   self.data and .target are required to be defined for use in Metrics()

        self.data = np.array(X[X.columns[1]])
        self.target = np.array(X[X.columns[1]])

        mean_value = X[X.columns[1]].mean()

        self.value_ = mean_value

    def predict(self, X):
        """Output model prediction.

        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return [self.value_] * len(X)

