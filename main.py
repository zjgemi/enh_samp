import numpy as np
from remd import REMD
from sym_dw import v, dv

betas = np.exp(np.arange(0, -3, -0.6))
print(betas)
nrep = len(betas)
remdobj = REMD(n=1, betas=betas, m=[1.0], lgam=1.0, pes=v, grad=dv)

xs = [[] for i in range(nrep)]
for i in range(100000):
    remdobj.integrator(dt=0.02)
    if i % 100 == 0:
        remdobj.exchange()
    for i in range(nrep):
        xs[i].append(remdobj.mdobjs[i].x[0])

import matplotlib.pyplot as plt
for i in range(nrep):
    plt.hist(xs[i], bins=100, label=str(betas[i]))
plt.legend()
plt.show()
