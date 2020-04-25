

# ...   https://dziganto.github.io/classes/data%20science/linear%20regression/
# ...   machine%20learning/object-oriented%20programming/python/
# ...   Understanding-Object-Oriented-Programming-Through-Machine-Learning/

import numpy as np


class Metrics :

    def __init__(self, X, y, model):
        self.data = X
        self.target = y
        self.model = model
        # degrees of freedom population dep. variable variance
        self._dft = X.shape[0] - 1
        # degrees of freedom population error variance
        self._dfe = X.shape[0] - X.shape[1] - 1

    def sse(self):
        
        """returns sum of squared errors (model vs actual)"""
        
        squared_errors = (self.target - self.predict(self.data)) ** 2
        self.sq_error_ = np.sum(squared_errors)
        
        return self.sq_error_

    def sst(self):
        
        """returns total sum of squared errors (actual vs avg(actual))"""
        
        avg_y = np.mean(self.target)
        squared_errors = (self.target - avg_y) ** 2
        self.sst_ = np.sum(squared_errors)
        return self.sst_

    def r_squared(self):
        
        """returns calculated value of r^2"""
        
        self.r_sq_ = 1 - self.sse( ) /self.sst()
        
        return self.r_sq_

    def adj_r_squared(self):
        
        """returns calculated value of adjusted r^2"""
        
        self.adj_r_sq_ = 1 - (self.sse( ) /self._dfe) / (self.sst( ) /self._dft)
        
        return self.adj_r_sq_

    def mse(self):
        
        """returns calculated value of mse"""
        
        self.mse_ = np.mean( (self.predict(self.data) - self.target) ** 2 )
        
        return self.mse_

    def print_metrics(self):
        
        """returns report of statistics for a given model object"""
        
        items = ( ('sse:', self.sse()), ('sst:', self.sst()),
                  ('mse:', self.mse()), ('r^2:', self.r_squared()),
                  ('adj_r^2:', self.adj_r_squared()))
        
        for item in items:
            print('{0:8} {1:.4f}'.format(item[0], item[1]))

