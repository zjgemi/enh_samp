import numpy as np
from remd import REMD
from its import ITS
from isremd import ISREMD
from sym_dw import v, dv
#from ho import v, dv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

beta = 1.0
betas = np.exp(np.arange(0, -3, -0.6))
nrep = len(betas)

remdobj = REMD(n=1, betas=betas, m=[1.0], lgam=1.0, pes=v, grad=dv)

xs = [[] for i in range(nrep)]
for i in range(100000):
    if i % 100 == 0:
        print(i,remdobj.mdobjs[0].x[0])
    remdobj.integrator(dt=0.02)
    if i % 100 == 0:
        remdobj.exchange()
    for i in range(nrep):
        xs[i].append(remdobj.mdobjs[i].x[0])

for i in range(nrep):
    plt.hist(xs[i], bins=100, density=True, label=str(betas[i]))
x0 = np.arange(-2,2,0.01)
y0 = np.exp(-beta*np.array(list(map(v, x0.reshape([-1,1])))))/0.644303
plt.plot(x0, y0, label='exact')
plt.legend()
plt.savefig("REMD.png")
plt.close()

#itsobj = ITS(n=1, beta=beta, betas=betas, ns=np.ones(nrep), m=[1.0], lgam=1.0, pes=v, grad=dv)
#
#xs = []
#ws = []
#for i in range(100000):
#    if i % 100 == 0:
#        print(i,itsobj.mdobj.x[0])
#    itsobj.integrator(dt=0.02)
#    samp = itsobj.mdobj.x
#    xs.append(samp[0])
#    ws.append(np.exp(-beta*(itsobj.pes(samp)-itsobj.effpes(samp))))
#
#plt.hist(xs, weights=ws, bins=100, density=True, label='ITS')
#x0 = np.arange(-2,2,0.01)
#y0 = np.exp(-beta*np.array(list(map(v, x0.reshape([-1,1])))))/0.644303
#plt.plot(x0, y0, label='exact')
#plt.legend()
#plt.savefig("ITS.png")
#plt.close()

#isremdobj = ISREMD(n=1, beta = beta, betas=betas, m=[1.0], lgam=1.0, infswap=True, nu=1, pes=v, grad=dv)
#
#xs = [[] for i in range(nrep)]
#ws = [[] for i in range(nrep)]
#for i in range(100000):
#    if i % 100 == 0:
#        print(i,isremdobj.mdobjs[0].x[0])
#    isremdobj.integrator(dt=0.02)
#    for j in range(nrep):
#        for i in range(nrep):
#            xs[j].append(isremdobj.mdobjs[i].x[0])
#            ws[j].append(isremdobj.eta[i,j])
#
#for i in range(nrep):
#    plt.hist(xs[i], weights=ws[i], bins=100, density=True, label=str(betas[i]))
#x0 = np.arange(-2,2,0.01)
#y0 = np.exp(-beta*np.array(list(map(v, x0.reshape([-1,1])))))/0.644303
#plt.plot(x0, y0, label='exact')
#plt.legend()
#plt.savefig("ISREMD_nuInf.png")
#plt.close()

