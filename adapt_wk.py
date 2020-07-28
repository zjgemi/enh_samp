import numpy as np
from its import ITS
from entr_barr import v, dv, alpha
#from sym_dw import v, dv

def weight(x,y):
    return np.sqrt(x*y)

if __name__ == "__main__":
    n = 2
    m = np.ones(n)
    beta = 1.0
    betas = np.arange(1.0, 0.4, -0.2)
    nrep = len(betas)
    lgam = 1.0
    dt = 0.1
    eqsteps = 10000
    nsteps = 100000
    nprint = 10000
    niter = 10

    nk = np.ones(nrep)#np.loadtxt("opt_wts.dat")
    itsobj = ITS(n=n, beta=beta, betas=betas, nk=nk, m=m, lgam=lgam, pes=v, grad=dv)

    mk = np.ones(nrep)
    Wk = np.zeros(nrep)
    for niter in range(niter):
        for i in range(eqsteps):
            itsobj.integrator(dt=dt)

        sumwk = np.zeros(nrep)
        for i in range(nsteps):
            itsobj.integrator(dt=dt)
            samp = itsobj.mdobj.x
            effpot = itsobj.effpes(samp)
            for k in range(nrep):
                sumwk[k] += np.exp(-betas[k]*itsobj.pes(samp)+beta*effpot)
            if i % nprint == nprint - 1:
                print(i,sumwk/sum(sumwk))

        pk = sumwk / sum(sumwk)
        for k in range(1,nrep):
            mk[k] = pk[k-1]/pk[k]
            #mk[k] = (weight(pk[k-1],pk[k])*pk[k-1]/pk[k] + Wk[k]*mk[k]) / (weight(pk[k-1],pk[k]) + Wk[k])
            #Wk[k] += weight(pk[k-1],pk[k])
            itsobj.nk[k] = itsobj.nk[k-1] * mk[k]

        print(niter,itsobj.nk)

    np.savetxt("opt_wts.dat", itsobj.nk)

