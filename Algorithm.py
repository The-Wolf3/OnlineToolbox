import numpy as np
import cvxpy as cp

class Algorithm:

    def __init__(self, n):
        self.n = n
        self.x = np.zeros((n,1))

    def update(self):
        pass
    
    def setX(self, x0):
        self.x = x0.reshape((self.n, 1))

    def eval(self, prob):
        return prob.f(self.x)

class MOSP(Algorithm):

    def __init__(self, n, alpha=0.3, omega=2):
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
        bigA = np.vstack((np.vstack((A, -A)),
                            np.zeros((2*(self.n-A.shape[0]), A.shape[1]))))
        bigb = np.vstack((np.vstack((b, -b)),
                            np.zeros((2*(self.n-b.shape[0]), 1))))
        updateVec = self.om*(bigA@self.x-bigb)
        self.dual = np.maximum(self.dual+updateVec, 0)
        grad = prob.grad(self.x)
        self.x += -self.alpha*(grad+bigA.T@self.dual)
        return self.x

class MOSPBasic(Algorithm):
    def __init__(self, n, alpha=0.2, omega=0.1):
        super().__init__(n)
        self.alpha = alpha
        self.om = omega
        self.dual = np.zeros((n, 1))

    def setDual(self, dual):
        self.dual = dual

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        bigA = np.vstack((-A, np.zeros((self.n-A.shape[0], A.shape[1]))))
        bigb = np.vstack((-b, np.zeros((self.n-b.shape[0], 1))))
        updateVec = self.om*(bigA@self.x-bigb)
        self.dual = np.maximum(self.dual+updateVec, 0)
        grad = prob.grad(self.x)
        self.x += -self.alpha*(grad+bigA.T@self.dual)
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


class LinConsProblem(Problem):

    def __init__(self, n=1):
        super().__init__(self)
        self.n = n
        self.A = np.zeros((0, n))
        self.b = np.zeros((0, 1))

    def getA(self):
        return self.A
    
    def getb(self):
        return self.b

    def setA(self, A):
        self.A = A

    def setb(self, b):
        self.b = b

def sigmoid(x):
    return 1/(1+np.e**-(x))

def sigmoidGrad(x):
    return sigmoid(x)**2 *np.e**-(x)

def sigmoidHess(x):
    return np.e**(-x)*(2*np.e**(-x)*sigmoid(x)**3)-sigmoidGrad(x)

class NetFlow(LinConsProblem):

    def __init__(self, n=1):
        super().__init__(n)
        self.f = self.loss
        self.grad = self.gradFct
        self.scaling = np.ones((n, 1))
        self.alpha = np.ones((n, 1))
        self.beta = np.zeros((n, 1))

    def setA(self, conjMatrix):
        pass

    def loss(self, x):
        return np.sum(self.scaling*
            (sigmoid(self.alpha*x+self.beta)+sigmoid(-self.alpha*x)))
    
    def gradFct(self, x):
        return self.alpha*(sigmoidGrad(self.alpha*x+self.beta)-sigmoidGrad(-self.alpha*x))

    def hessFct(self, x):
        return self.alpha**2*(sigmoidHess((self.alpha*x+self.beta)*np.identity(self.n))+
                sigmoidHess(-self.alpha*x*np.identity(self.n)))
    
    def getH(self, x):
        return self.hessFct(x)

class ConvNetFlow(LinConsProblem):

    def __init__(self, n=1):
        super().__init__(n)
        self.f = self.loss
        self.grad = self.gradFct
        self.c = np.ones((n, 1))
        self.alpha = np.ones((n, 1))
        self.beta = np.zeros((n, 1))

    def setLossParams(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def loss(self, x):
        return np.sum(self.alpha*x**2+self.beta*x+self.c)
    
    def gradFct(self, x):
        return self.alpha*2*x+self.beta

    def hessFct(self, x):
        return np.identity(self.n) *2* self.alpha
    
    def getH(self, x):
        return self.hessFct(x)

    def optimal(self):
        x = cp.Variable((self.n, 1))
        cost = cp.sum(cp.multiply(self.alpha, x**2)+cp.multiply(self.beta,x)+self.c)
        constraints = []
        for row in range(self.A.shape[0]):
            constraints.append(cp.sum(cp.multiply(self.A[row].reshape(self.n,1), x)) == self.b[row])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver="GUROBI")
        return prob.value, x.value





class QLCP(Problem):

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





