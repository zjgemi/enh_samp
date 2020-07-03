def v(x):
    return 8.0*(1.0-x[0]**2)**2

def dv(x):
    return 2*8.0*(x[0]**2-1.0)*2*x[0]
