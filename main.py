import numpy as np
from md import MD
from remd import REMD
from its import ITS
from isremd import ISREMD
from entr_barr import v, dv, alpha
#from sym_dw import v, dv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

if __name__ == "__main__":
    n = 2
    m = [1.0, 1.0]
    beta = 1.0
    betas = np.exp(np.arange(0, -3, -0.6))
    nrep = len(betas)
    lgam = 1.0
    dt = 0.02
    nsteps = 1000000
    nprint = 1000
    jobtype = "REMD"
    figname = "REMD"
    
    if jobtype == "MD":
        mdobj = MD(n=n, beta=beta, m=m, lgam=lgam, pes=v, grad=dv)
        xs = []
        for i in range(nsteps):
            if i % nprint == 0:
                print(i,mdobj.x[0])
            mdobj.lfmiddle_integrator(dt=dt)
            xs.append(mdobj.x[0])
        ws = np.ones_like(xs)
    
    elif jobtype == "REMD":
        remdobj = REMD(n=n, betas=betas, m=m, lgam=lgam, pes=v, grad=dv)
        xs = []
        for i in range(nsteps):
            if i % nprint == 0:
                print(i,remdobj.mdobjs[0].x[0])
            remdobj.integrator(dt=dt)
            if i % 1 == 0:
                remdobj.exchange()
            xs.append(remdobj.mdobjs[0].x[0])
        ws = np.ones_like(xs)
    
    elif jobtype == "ITS":
        itsobj = ITS(n=n, beta=beta, betas=betas, ns=np.ones(nrep), m=m, lgam=lgam, pes=v, grad=dv)
        xs = []
        ws = []
        for i in range(nsteps):
            if i % nprint == 0:
                print(i,itsobj.mdobj.x[0])
            itsobj.integrator(dt=dt)
            samp = itsobj.mdobj.x
            xs.append(samp[0])
            ws.append(np.exp(-beta*(itsobj.pes(samp)-itsobj.effpes(samp))))
    
    elif jobtype == "ISREMD":
        isremdobj = ISREMD(n=n, beta=beta, betas=betas, m=m, lgam=lgam, infswap=True, nu=1, pes=v, grad=dv)
        xs = []
        ws = []
        for i in range(nsteps):
            if i % nprint == 0:
                print(i,isremdobj.mdobjs[0].x[0])
            isremdobj.integrator(dt=dt)
            for i in range(nrep):
                xs.append(isremdobj.mdobjs[i].x[0])
                ws.append(isremdobj.eta[i,0])
    
    dens, edges = np.histogram(xs, weights=ws, range=(-2,2), bins=100, density=True)
    cents = (edges[:-1] + edges[1:])/2
    fes = -np.log(dens)/beta
    plt.plot(cents, fes-min(fes), label=jobtype)
    x0 = np.arange(-2,2,0.01)
    y0 = x0**4 - (n-1)*alpha*x0**2/2/beta
    plt.plot(x0, y0-min(y0), label='exact')
    plt.legend()
    plt.savefig(figname+".png")
    plt.close()
    
