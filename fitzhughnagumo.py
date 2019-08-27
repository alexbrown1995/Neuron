from numpy import *
# Import the ODE integrator
from scipy.integrate import odeint
# Get access to the root-finder
from scipy.optimize import fsolve
# Plotting
from matplotlib.pyplot import *
#from scipy.optimize import lmfit
from lmfit import Model


def fhn(V,t,a,b,c,I):
    return [V[0]*(a-V[0])*(V[0]-1)-V[1]+I, b*V[0]-c*V[1]]

a=0.2
b=0.02
c=0.02
I=0.59
V0=[0,0]
t=linspace(0,300,1000)


V=odeint(fhn, V0, t, args=(a,b,c,I,))

j = 0
v=[]
w=[]
while (j <= 999):

    v.append(V[j][0])
    w.append(V[j][1])
    j = j + 1


plot(t,v)
show()