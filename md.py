import numpy as np

class MD:
    def __init__(self, n=1, beta=1.0,
            m=None, x=None, p=None,
            lgam=1.0,
            pes=None, grad=None):
        self.n = n # DOF
        self.beta = beta # 1/(k_B*T)
        self.lgam = lgam # friction coefficient in Langevin

        # initialize mass
        if m == None:
            self.m = np.ones(n)
        else:
            m = np.array(m)
            assert(m.shape == (n,))
            self.m = m

        # initialize position
        if x == None:
            self.x = np.zeros(n)
        else:
            x = np.array(x)
            assert(x.shape == (n,))
            self.x = x

        # initialize momentum
        if p == None:
            self.p = np.random.normal(loc=0.0,scale=1.0,size=n)*np.sqrt(self.m/self.beta)
        else:
            p = np.array(p)
            assert(p.shape == (n,))
            self.p = p

        if pes != None:
            self.pes = pes

        if grad != None:
            self.grad = grad

        self.f = -self.grad(self.x)

    def pes(self, x):
        # potential energy surface
        # output is a scalar
        pass

    def grad(self, x):
        # gradient of the PES
        # output is n-dim vector
        pass

    def lfmiddle_integrator(self, dt):
        self.p += self.f * dt
        self.x += self.p / self.m * dt/2
        c1 = np.exp(-self.lgam*dt)
        c2 = np.sqrt(1.0-c1**2)
        eta = np.random.normal(loc=0.0,scale=1.0,size=self.n)*np.sqrt(self.m/self.beta)
        self.p = c1 * self.p + c2 * eta
        self.x += self.p / self.m * dt/2
        self.f = -self.grad(self.x)

