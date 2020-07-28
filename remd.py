import numpy as np
from md import MD

class REMD:
    def __init__(self, n=1, beta=1.0, betas=np.array([1.0]),
            m=None, x=None, p=None,
            lgam=1.0,
            pes=None, grad=None):
        self.n = n # DOF
        betas = np.array(betas)
        self.nrep = len(betas) # number of replicas
        self.betas = betas # list of 1/(k_B*T)

        if pes is not None:
            self.pes = pes

        if grad is not None:
            self.grad = grad

        self.mdobjs = [MD(n=n, beta=beta,
            m=m if m is not None else None,
            x=x[i,:] if x is not None else None,
            p=p[i,:] if p is not None else None,
            lgam=lgam, pes=self.wrap(self.pes,betas[i]/beta),
            grad=self.wrap(self.grad,betas[i]/beta))
            for i in range(self.nrep)]

    def wrap(self, func, scale):
        def f(*args, **kwargs):
            return func(*args, **kwargs)*scale
        return f

    def pes(self, x):
        # potential energy surface
        # output is a scalar
        pass

    def grad(self, x):
        # gradient of the PES
        # output is n-dim vector
        pass

    def integrator(self, dt):
        for i in range(self.nrep):
            self.mdobjs[i].lfmiddle_integrator(dt)

    def exchange(self):
        i = int(np.random.random()*(self.nrep-1))

        diff = -(self.betas[i]-self.betas[i+1])*(self.pes(self.mdobjs[i+1].x)-self.pes(self.mdobjs[i].x))

        if diff >= 0.0 or np.random.random() < np.exp(diff):
            tmp = self.mdobjs[i].x
            self.mdobjs[i].x = self.mdobjs[i+1].x
            self.mdobjs[i+1].x = tmp
            return 1.0
        else:
            return 0.0

