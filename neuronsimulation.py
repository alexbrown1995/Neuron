from numpy import *
# Import the ODE integrator
from scipy.integrate import odeint
# Get access to the root-finder
from scipy.optimize import fsolve
# Plotting
from matplotlib.pyplot import *

L=1e-7
C1=1e-6
R1=3.3e3
I=0.3e-3
E=10


#y = np.array([0,7.25, 7.5, 8, 8.25, 8.45, 8.4626, 8.45, 8.25, 8.0625, 8, 7.84375, 7.71875, 7.68125, 7.75, 7.9125, 8, 8])
#x= np.log(np.array([0.1e-272,0.02e-3, 0.056818182e-3, 0.102272727e-3, 0.107954545e-3, 0.15625e-3, 0.3125e-3, 0.5e-3, 0.539772727e-3, 0.625e-3, 0.738636364e-3, 0.965909091e-3,1.25e-3, 1.647727273e-3, 2.5e-3, 5e-3, 6.079545455e-3, 6.704545455e-3]))

x=np.array([0.096153846e-3,0.144230769e-3,0.211538462e-3,0.326923077e-3,0.5e-3,1e-3,1.692307692e-3,2e-3,2.5e-3,2.788461538e-3,2.865384615e-3])
#x=np.array([0.096153846,0.144230769,0.211538462,0.326923077,0.5,1,1.692307692,2,2.5,2.788461538,2.865384615])
y=np.array([7,7.58,8,8.5,8.388888889,7.875,7.5,7.611111111,8.208333333,8.5,8.611111111])
def polynomialeval(x,y,xval,deg):
    pvals=np.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
    p=0
    for i in range(0,deg+1):
        p=p+pvals[i]*(xval**(deg-i))
    return p

def Vec(ln):
    v=polynomialeval(x,y,ln,3)
    return v

pvals=np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)

def neuronode(V,t,C1,R1,E,L,I):
    return [(1/C1)*(((E-V[0])/R1)-V[1]+I),(1/L)*(V[0]-Vec(V[1]))]#(pvals[0]*V[1]**3+pvals[1]*V[1]**2+pvals[2]*V[1]+pvals[3]))]


V0=[0,0]
T=20e-3
t=linspace(0,T,1000)

for i in range (0,10):
    L=1e-7+i*1e-7

    V=odeint(neuronode, V0, t, args=(C1,R1,E,L,I,))
    j = 0
    V1=[]
    In=[]
    while (j <= 999):


        V1.append(V[j][0])
        In.append(V[j][1])
        j = j + 1

    print(V)
    #plot(t,V1)
    plot(t,In)
    show()
    plot(t,V1)
    show()
print(Vec(2.865384615e-3))
#(pvals[0]*V[1]**3+pvals[1]*V[1]**2+pvals[2]*V[1]+pvals[3])+I)