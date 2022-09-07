import numpy as np

class Algorithm:

    def __init__(self, n):
        self.n = n
        self.x = np.zeros(n)

    def update(self):
        pass
    
    def setX(self, x0):
        self.x = x0



class OPENM(Algorithm):

    def __init__(self):
        super().__init__()