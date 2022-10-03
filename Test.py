from Algorithm import *


# prob = QLCP(10)
# Newton = OPENM(10)
# Mosp = MOSP(10)
# # A = prob.getA()
# # b = prob.getb()
# # x = np.ones((10,1))
# # x += A.T@np.linalg.inv(A@A.T)@(b-A@x)
# # print(np.linalg.norm(A@x-b))
# print("Condition number: {}".format(np.linalg.cond(prob.getA())))
# print(Newton.eval(prob))
# print(Newton.violation(prob))

# for i in range(1, 5):
#     print("iteration {}".format(i))
#     print(Newton.update(prob))
#     print(Newton.eval(prob))
#     print(Newton.violation(prob))

Newton = OPENM(6)
mosp = MOSPBasic(5)
prob2 = ConvNetFlow(6)
#Newton.setX(np.array([42, 11, 22, 7, 20.]))

A = np.array([[-1,1,1,0,0,0],[0,-1,0,1,0,0],[0,0,-1,0,1,0],[0,0,0,-1,-1,0]], dtype='float64')
b = np.array([-10, -5, -4, -25], dtype='float64').reshape(4,1)
prob2.setA(A)
prob2.setb(b)
prob2.alpha = np.array([[1],[1],[1],[1000],[3],[3]])
prob2.beta = np.ones((6,1))



print(prob2.optimal())
# print(mosp.eval(prob2))
# print(mosp.violation(prob2))
# for i in range(8):
#     print("iteration {}".format(i))
#     mosp.update(prob2)
#     print(mosp.x)
#     print(mosp.eval(prob2))
#     print(mosp.violation(prob2))

print(Newton.eval(prob2))
print(Newton.violation(prob2))
for i in range(3):
    print("iteration {}".format(i))
    Newton.update(prob2)
    print(Newton.x)
    print(Newton.eval(prob2))
    print(Newton.violation(prob2))


"""ax = np.linspace(-6,50,1000)
fct = []
grad = []
hess = []
for i in ax:
    x = np.ones((2,1))*i
    fct.append(prob2.f(x))
    grad.append(prob2.grad(x)[0])
    hess.append(prob2.getH(x)[0,0])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(ax, fct)
plt.plot(ax, grad)
plt.plot(ax, hess)
plt.show() """
