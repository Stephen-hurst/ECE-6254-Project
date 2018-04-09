import numpy as np

class LookupTable():
    '''Implement an ML model with .predict and .fit as a simple lookup table'''
    def __init__(self, max_states=8192):
        self._table = np.zeros(max_states)
        
    def fit(self, X, Y):
        X_lookup = np.sum(X*np.array([(1<<7), (1<<2), 2, 1]), 1, dtype='int')
        self._table[X_lookup] = Y
            
    def predict(self, X):
        X_lookup = np.sum(X*np.array([(1<<7), (1<<2), 2, 1]), 1, dtype='int')
        return self._table[X_lookup]

