
# ...   https://dziganto.github.io/classes/data%20science/linear%20regression/
# ...   machine%20learning/object-oriented%20programming/python/
# ... Understanding-Object-Oriented-Programming-Through-Machine-Learning/

class LinearRegression(Metrics):

    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit model coefficients.

        Arguments:
        X: 1D or 2D numpy array
        y: 1D numpy array
        """

        # training data & ground truth data
        self.data = X
        self.target = y

        # degrees of freedom population dep. variable variance
        self._dft = X.shape[0] - 1
        # degrees of freedom population error variance
        self._dfe = X.shape[0] - X.shape[1] - 1

        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # add bias if fit_intercept
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # closed form solution
        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

    def predict(self, X):
        """Output model prediction.

        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + np.dot(X, self.coef_)