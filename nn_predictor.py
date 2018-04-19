from keras.models import Sequential
from keras.layers import Dense, Activation, Input

# class LinearPredictor(Sequential):
#     def __init__(self, inputs):
#         Sequential.__init__(self)
#         self.add(Dense(units=1, activation='linear', input_dim=inputs, bias_initializer='zeros', kernel_initializer='zeros'))
#         self.compile(loss='mean_squared_error',optimizer='sgd', metrics=[])

class NNPredictor(Sequential):
    def __init__(self, inputs):
        Sequential.__init__(self)
        self.add(Dense(units=128, activation='relu', input_dim=inputs))
        self.add(Dense(units=64, activation='relu'))
        self.add(Dense(units=16, activation='relu'))
        self.add(Dense(units=1, activation='linear'))
        self.compile(loss='mean_squared_error',optimizer='sgd', metrics=[])