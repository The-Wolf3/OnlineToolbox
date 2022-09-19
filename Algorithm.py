import numpy as np

class Algorithm:

    def __init__(self, n):
        self.n = n
        self.x = np.ones((n,1))

    def update(self):
        pass
    
    def setX(self, x0):
        self.x = x0

    def eval(self, prob):
        return prob.f(self.x)

class MOSP(Algorithm):

    def __init__(self, n, alpha=1, omega=1):
        super().__init__(n)
        self.alpha = alpha
        self.om = omega
        self.dual = np.zeros((2*n, 1))

    def setDual(self, dual):
        self.dual = dual

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        bigA = np.hstack((np.hstack((A, -A)),
                            np.zeros((2*self.n-A.shape[0], A.shape[1]))))
        bigb = np.hstack((np.hstack((b, -b)),
                            np.zeros((self.n-b.shape[0], 1))))
        updateVec = self.om*(bigA@self.x-bigb)
        self.dual = np.maximum(self.dual+updateVec, 0)
        grad = prob.grad(self.x)
        self.x += -self.alpha*(grad+self.dual.T@bigA)
        return self.x


class OPENM(Algorithm):

    def __init__(self, n):
        super().__init__(n)

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        self.x += A.T@np.linalg.inv(A@A.T)@(b-A@self.x)
        grad = prob.grad(self.x)
        H = prob.getH(self.x)
        D = np.vstack((np.hstack((H, np.transpose(A))), 
                np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
        augGrad = np.vstack((grad, np.zeros((A.shape[0],1))))
        delta = (np.linalg.inv(D)@augGrad)[:self.n]
        self.x -= delta
        print(np.linalg.norm(delta))
        return self.x

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)


class Problem:

    def __init__(self, f=None, grad=None):
        self.f = f
        self.grad = grad

    def increment(self):
        pass
    def optimal(self):
        pass
    def setF(self, f):
        self.f = f
    def setGrad(self, grad):
        self.grad = grad

class QLCP:

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.increment()

    def increment(self):
        Q = np.random.random((self.n, self.n))
        Q = (Q + Q.T)/2
        def f(x):
            return float(x.T@Q@x)
        def grad(x):
            return 2*Q@x
        self.f = f
        self.grad = grad
        self.H = Q
        m = np.random.randint(1,5)
        self.A = np.random.random((m, self.n))
        self.b = np.random.random((m, 1))
    
    def getA(self):
        return self.A
    def getb(self):
        return self.b
    def getH(self, x):
        return self.H


prob = QLCP(10)
Newton = OPENM(10)
Mosp = MOSP(10)
# A = prob.getA()
# b = prob.getb()
# x = np.ones((10,1))
# x += A.T@np.linalg.inv(A@A.T)@(b-A@x)
# print(np.linalg.norm(A@x-b))
print("Condition number: {}".format(np.linalg.cond(prob.getA())))
print(Newton.eval(prob))
print(Newton.violation(prob))

for i in range(1, 5):
    print("iteration {}".format(i))
    print(Newton.update(prob))
    print(Newton.eval(prob))
    print(Newton.violation(prob))