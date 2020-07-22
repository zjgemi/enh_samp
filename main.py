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
from multiprocessing import Pool
from functools import partial

def run_trj(itrj, jobname, n, m, beta, betas, lgam, dt, eqsteps, nsteps, pes, grad, nprint=1000):
    np.random.seed(itrj)
    nrep = len(betas)
    jobtype = jobname
    
    if jobtype == "MD":
        mdobj = MD(n=n, beta=beta, m=m, lgam=lgam, pes=pes, grad=grad)

        print("Equilibrating..")
        for i in range(eqsteps):
            mdobj.lfmiddle_integrator(dt=dt)

            if i % nprint == 0:
                print(i,mdobj.x[0])

        xs = []
        print("Sampling..")
        for i in range(nsteps):
            mdobj.lfmiddle_integrator(dt=dt)
            xs.append(mdobj.x[0])

            if i % nprint == 0:
                print(i,mdobj.x[0])

        ws = np.ones(len(xs))
    
    elif jobtype == "REMD":
        remdobj = REMD(n=n, beta=beta, betas=betas, m=m, lgam=lgam, pes=pes, grad=grad)
        sumratio = 0
        cnt = 0

        print("Equilibrating..")
        for i in range(eqsteps):
            remdobj.integrator(dt=dt)
            if i % 1 == 0:
                sumratio += remdobj.exchange()
                cnt += 1

            if i % nprint == 0:
                print(i,remdobj.mdobjs[0].x[0],sumratio/cnt)

        xs = []
        print("Sampling..")
        for i in range(nsteps):
            remdobj.integrator(dt=dt)
            if i % 1 == 0:
                sumratio += remdobj.exchange()
                cnt += 1
            xs.append(remdobj.mdobjs[0].x[0])

            if i % nprint == 0:
                print(i,remdobj.mdobjs[0].x[0],sumratio/cnt)

        ws = np.ones(len(xs))
    
    elif jobtype == "ITS":
        nk = np.loadtxt("opt_wts.dat")
        itsobj = ITS(n=n, beta=beta, betas=betas, nk=nk, m=m, lgam=lgam, pes=pes, grad=grad)

        print("Equilibrating..")
        for i in range(eqsteps):
            itsobj.integrator(dt=dt)

            if i % nprint == 0:
                print(i,itsobj.mdobj.x[0])

        xs = []
        ws = []
        print("Sampling..")
        for i in range(nsteps):
            itsobj.integrator(dt=dt)
            samp = itsobj.mdobj.x
            wt = np.exp(-beta*(itsobj.pes(samp)-itsobj.effpes(samp)))
            xs.append(samp[0])
            ws.append(wt)

            if i % nprint == 0:
                print(i,itsobj.mdobj.x[0])

    elif jobtype == "ISREMD":
        isremdobj = ISREMD(n=n, beta=beta, betas=betas, m=m, lgam=lgam, nu=200, pes=pes, grad=grad)

        print("Equilibrating..")
        for i in range(eqsteps):
            isremdobj.integrator(dt=dt)

            if i % nprint == 0:
                print(i,isremdobj.mdobjs[0].x[0])

        xs = []
        ws = []
        for i in range(nsteps):
            isremdobj.integrator(dt=dt)
            for i in range(nrep):
                samp = isremdobj.mdobjs[i].x
                wt = isremdobj.eta[i,0]
                xs.append(samp[0])
                ws.append(wt)

            if i % nprint == 0:
                print(i,isremdobj.mdobjs[0].x[0])

    avgx0 = sum(np.array(xs)*np.array(ws)) / sum(np.array(ws))
    with open("estimators_"+str(itrj)+".dat", "w") as f:
        f.write(str(avgx0)+"\n")

    dens, edges = np.histogram(xs, weights=ws, range=(-2,2), bins=100, density=True)
    cents = (edges[:-1] + edges[1:])/2
    np.savetxt("dens_"+str(itrj)+".dat", np.stack([cents, dens], axis=1))

if __name__ == "__main__":
    ntrjs = 16
    pool = Pool(ntrjs)

    jobnames = ["MD", "REMD", "ITS", "ISREMD"]
    nstepss = [15000000, 1000000, 15000000, 1000000]
    n = 41
    m = np.ones(n)
    beta = 40.0
    betas = np.arange(40.0, 10.0, -2.0)
    lgam = 1.0
    dt = 0.05
    eqsteps = 10000
    nprint = 1000

    for jobname, nsteps in zip(jobnames,nstepss):
        pool.map(partial(run_trj, jobname=jobname, n=n, m=m, beta=beta, betas=betas, lgam=lgam, dt=dt, eqsteps=eqsteps, nsteps=nsteps, pes=v, grad=dv, nprint=nprint), range(ntrjs))

        dat = []
        for itrj in range(ntrjs):
            dat.append(np.loadtxt("estimators_"+str(itrj)+".dat"))
        dat = np.array(dat)
        with open("estimators_"+jobname+".dat", "w") as f:
            f.write(str(np.mean(dat))+" "+str(np.std(dat)/np.sqrt(ntrjs-1))+"\n")

        dens = []
        for itrj in range(ntrjs):
            dat = np.loadtxt("dens_"+str(itrj)+".dat")
            x = dat[:,0]
            dens.append(dat[:,1])
        dens = np.array(dens)
        np.savetxt("fes_"+jobname+".dat", np.stack([x, -np.log(np.mean(dens,0))/beta, np.std(dens,0)/np.sqrt(ntrjs-1)/np.mean(dens,0)/beta], axis=1))

