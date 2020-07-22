import numpy as np
from md import MD

class ITS:
    def __init__(self, n=1, beta=1.0, betas=[1.0],
            nk=[1.0], m=None, x=None, p=None,
            lgam=1.0, pes=None, grad=None):
        self.n = n # DOF
        self.beta = beta # 1/(k_B*T)
        self.betas = betas
        self.nk = nk
        self.nbeta = len(betas)

        if pes is not None:
            self.pes = pes

        if grad is not None:
            self.grad = grad

        self.mdobj = MD(n=n, beta=beta, m=m, x=x,
            p=p, lgam=lgam, pes=self.effpes, grad=self.effgrad)

    def pes(self, x):
        # potential energy surface
        # output is a scalar
        pass

    def grad(self, x):
        # gradient of the PES
        # output is n-dim vector
        pass

    def effpes(self, x):
        den = 0
        for i in range(self.nbeta):
            den += self.nk[i]*np.exp(-self.betas[i]*self.pes(x))
        return -np.log(den)/self.beta

    def effgrad(self, x):
        num = 0
        den = 0
        for i in range(self.nbeta):
            num += self.nk[i]*self.betas[i]*np.exp(-self.betas[i]*self.pes(x))
            den += self.nk[i]*np.exp(-self.betas[i]*self.pes(x))
        return num/self.beta/den*self.grad(x)

    def integrator(self, dt):
        self.mdobj.lfmiddle_integrator(dt)

