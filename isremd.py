import numpy as np
from itertools import permutations
from md import MD

class ISREMD:
    def __init__(self, n=1, beta=1.0, betas=np.array([1.0]),
            m=None, x=None, p=None,
            lgam=1.0, infswap=False, nu=1e4,
            pes=None, grad=None):
        self.n = n # DOF
        self.beta = beta
        betas = np.array(betas)
        self.nrep = len(betas) # number of replicas
        self.betas = betas # list of 1/(k_B*T)
        self.infswap = infswap
        self.nu = nu

        self.sigmas = list(range(self.nrep))

        self.mdobjs = [MD(n=n, beta=beta,
            m=m if m is not None else None,
            x=x[i,:] if x is not None else None,
            p=p[i,:] if p is not None else None,
            lgam=lgam, pes=pes, grad=grad)
            for i in range(self.nrep)]

        if pes is not None:
            self.pes = pes

        if grad is not None:
            self.grad = grad

        self.factor = None

    def pes(self, x):
        # potential energy surface
        # output is a scalar
        pass

    def grad(self, x):
        # gradient of the PES
        # output is n-dim vector
        pass

    def get_factors(self, dt):
        pots = [self.pes(self.mdobjs[i].x) for i in range(self.nrep)]

        if self.infswap:
            sumrho = 0
            self.eta = np.zeros([self.nrep, self.nrep])
            for sigma in permutations(range(self.nrep)):
                sigma = list(sigma)
                rho = np.exp(-sum(self.betas[sigma]*pots))
                sumrho += rho
                for i, j in enumerate(sigma):
                    self.eta[i,j] += rho
            self.eta /= sumrho
        else:
            t = 0
            self.eta = np.zeros([self.nrep, self.nrep])

            # initialize rate matrix
            rates = np.zeros([self.nrep, self.nrep])
            for i in range(self.nrep):
                for j in range(i+1, self.nrep):
                    diff = (self.betas[self.sigmas[j]]-self.betas[self.sigmas[i]])*(pots[j]-pots[i])
                    rates[i,j] = self.nu*np.exp(diff)

            cnt = 0
            while t < dt:
                cnt += 1
                sumrate = np.sum(rates)
                tau = -np.log(np.random.random())/sumrate

                if t + tau > dt:
                    tau = dt - t

                for i in range(self.nrep):
                    self.eta[i,self.sigmas[i]] += tau / dt # TODO: more efficient?

                rn = np.random.random()
                sump = 0
                flag = False
                for i in range(self.nrep):
                    for j in range(i+1, self.nrep):
                        sump += rates[i,j] / sumrate
                        if sump > rn:
                            flag = True
                            break
                    if flag:
                        break

                tmp = self.sigmas[i]
                self.sigmas[i] = self.sigmas[j]
                self.sigmas[j] = tmp
                t += tau

                # update rate matrix
                for k in range(i):
                    diff = (self.betas[self.sigmas[i]]-self.betas[self.sigmas[k]])*(pots[i]-pots[k])
                    rates[k,i] = self.nu*np.exp(diff)

                for k in range(i+1,self.nrep):
                    diff = (self.betas[self.sigmas[k]]-self.betas[self.sigmas[i]])*(pots[k]-pots[i])
                    rates[i,k] = self.nu*np.exp(diff)

                for k in range(j):
                    diff = (self.betas[self.sigmas[j]]-self.betas[self.sigmas[k]])*(pots[j]-pots[k])
                    rates[k,j] = self.nu*np.exp(diff)

                for k in range(j+1,self.nrep):
                    diff = (self.betas[self.sigmas[k]]-self.betas[self.sigmas[j]])*(pots[k]-pots[j])
                    rates[j,k] = self.nu*np.exp(diff)

            print(cnt)

        self.factor = np.zeros(self.nrep)
        for j in range(self.nrep):
            self.factor[j] = sum(self.betas*self.eta[:,j])/self.beta

    def integrator(self, dt):
        if self.factor is None:
            self.get_factors(dt)
            for i in range(self.nrep):
                self.mdobjs[i].f *= self.factor[i]

        for i in range(self.nrep):
            self.mdobjs[i].lfmiddle_integrator(dt)

        self.get_factors(dt)
        for i in range(self.nrep):
            self.mdobjs[i].f *= self.factor[i]

