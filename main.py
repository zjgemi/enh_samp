import numpy as np
from remd import REMD
from its import ITS
from sym_dw import v, dv

beta = 1.0
betas = np.exp(np.arange(0, -3, -0.6))
print(betas)
nrep = len(betas)

#remdobj = REMD(n=1, betas=betas, m=[1.0], lgam=1.0, pes=v, grad=dv)
#
#xs = [[] for i in range(nrep)]
#for i in range(100000):
#    remdobj.integrator(dt=0.02)
#    if i % 100 == 0:
#        remdobj.exchange()
#    for i in range(nrep):
#        xs[i].append(remdobj.mdobjs[i].x[0])
#
#import matplotlib.pyplot as plt
#for i in range(nrep):
#    plt.hist(xs[i], bins=100, density=True, label=str(betas[i]))
#x0 = np.arange(-2,2,0.01)
#y0 = np.exp(-beta*np.array(list(map(v, x0.reshape([-1,1])))))/0.644303
#plt.plot(x0, y0, label='exact')
#plt.legend()
#plt.show()

itsobj = ITS(n=1, beta=beta, betas=betas, ns=np.ones(nrep), m=[1.0], lgam=1.0, pes=v, grad=dv)

xs = []
ws = []
for i in range(100000):
    itsobj.integrator(dt=0.02)
    samp = itsobj.mdobj.x
    xs.append(samp[0])
    ws.append(np.exp(-beta*(itsobj.pes(samp)-itsobj.effpes(samp))))

import matplotlib.pyplot as plt
plt.hist(xs, weights=ws, bins=100, density=True, label='ITS')
x0 = np.arange(-2,2,0.01)
y0 = np.exp(-beta*np.array(list(map(v, x0.reshape([-1,1])))))/0.644303
plt.plot(x0, y0, label='exact')
plt.legend()
plt.show()
