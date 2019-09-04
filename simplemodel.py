from numpy import *
# Import the ODE integrator
import scipy
from scipy import linalg
from scipy.integrate import odeint
# Get access to the root-finder
from scipy.optimize import fsolve
# Plotting
from matplotlib.pyplot import *
import csv
from mpl_toolkits import mplot3d
L=1e-7
C1=1e-6
C2=10e-6
C3=2e-6
R1=3.3e3
R3=300e3
R4=10e3
I=0.7e-3
E=10

x=np.array([0.096153846e-3,0.144230769e-3,0.211538462e-3,0.326923077e-3,0.5e-3,1e-3,1.692307692e-3,2e-3,2.5e-3,2.788461538e-3,2.865384615e-3])
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
s=-1
def neuronode(V,t,L,E,R1,R3,C2):

    return [ V1(V[0],V[2],V[3],R1,R3,R4,I,E,C1,C2,s),(1/C2)*(switch(V[2],V[0],V[3],R1,R3,I,s)),1/C3*(V3(V[2],R4,V[0],R3,s)),(1/L)*(V[0] - Vec(V[3]))]



def V1(V1,V3,In,R1,R3,R4,I,E,C1,C2,s):


    if (s==1):
        out=(1/(C1+C2))*(((E-V1)/R1)-In-((V1-V3)/R3)+I)
    else:
        out=1/C1*(((E-V1)/R1)-In-((V1-V3)/R3)+I)
    return out


def switch(V3,V1,In,R1,R3,I,s):

    if (s==1):
        out=(1/(1+(C2/C1)))*(((E-V1)/R1)-In-((V1-V3)/R3)+I)
    else:
        out=0
    return out

def V3(V,R4,V1,R3,s):

    if s==-1:
        if (V>=0.65):
            s=1
            out=-(((V1-V)/R3))#-((V-0.65)/R4)
        else:
            out=(V1-V)/R3
    elif s==1:
        if (V >= 0):

            out = -((V1 - V) / R3)# - ((V - 0.65) / R4)
        else:
            s=-1
            out = (V1 - V) / R3

    return out



L=1e-7
V0=[0,0,0,0]
T=300e-3
t=linspace(0,T,1000)

V = odeint(neuronode, V0, t, args=(L,E,R1,R3,C2,))
j = 0
v1 = []
vc2=[]
v3 = []
In=[]
while (j <= 999):
    v1.append(V[j][0])
    vc2.append(V[j][1])
    v3.append(V[j][2])
    In.append(V[j][3])
    j = j + 1

plot(t,v1,'b')
plot(t,vc2,'g')
plot(t,v3,'r')
plot(t,In,'c')
show()