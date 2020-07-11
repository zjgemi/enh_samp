import numpy as np
from md import MD

class REMD:
    def __init__(self, n=1, betas=np.array([1.0]),
            m=None, x=None, p=None,
            lgam=1.0,
            pes=None, grad=None):
        self.n = n # DOF
        betas = np.array(betas)
        self.nrep = len(betas) # number of replicas
        self.betas = betas # list of 1/(k_B*T)

        self.mdobjs = [MD(n=n, beta=betas[i],
            m=m if m != None else None,
            x=x[i,:] if x != None else None,
            p=p[i,:] if p != None else None,
            lgam=lgam, pes=pes, grad=grad)
            for i in range(self.nrep)]

        if pes != None:
            self.pes = pes

        if grad != None:
            self.grad = grad

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
        for i in list(range(0,self.nrep-1,2)) + list(range(1,self.nrep-1,2)):
            diff = -self.betas[i]*(self.pes(self.mdobjs[i+1].x)-self.pes(self.mdobjs[i].x))
            -self.betas[i+1]*(self.pes(self.mdobjs[i].x)-self.pes(self.mdobjs[i+1].x))

            if diff > 0.0:
                prob = 1.0
            else:
                prob = np.exp(diff)

            if prob >= 1.0 or np.random.random() < prob:
                tmp = self.mdobjs[i].x
                self.mdobjs[i].x = self.mdobjs[i+1].x
                self.mdobjs[i+1].x = tmp

