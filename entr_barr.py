import numpy as np

alpha = 4.0

def v(x):
    x = np.array(x)
    return x[0]**4 + np.exp(-alpha*x[0]**2)*sum(x[1:]**2)/2

def dv(x):
    x = np.array(x)
    return np.concatenate([[4*x[0]**3-alpha*x[0]*np.exp(-alpha*x[0]**2)*sum(x[1:]**2)], np.exp(-alpha*x[0]**2)*x[1:]])
