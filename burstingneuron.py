from numpy import *
# Import the ODE integrator
from scipy.integrate import odeint
# Get access to the root-finder
from scipy.optimize import fsolve
# Plotting
from matplotlib.pyplot import *

L=1e-7
C1=1e-6
C2=10e-6
C3=2e-6
R1=3.3e3
R3=100e3
R4=10e3
I=0.5e-3
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

def Vce(I2,I4):
    aI=-(I2+I4)/I2
    aN=-I2/(I2+I4)
    return -0.026*log(aI*(1-(I2/I4)*((1-aN)/aN))/(1+(I2/I4)*(1-aI)))

#def neuronode(V,t,L,E,C1,C3,R1,R3,R4,I):
#    return [(1/C1)*(((E-V[0])/R1)-V[2]-V[3]-((V[0]-V[1])/R3)+I),(1/C3)*(((V[0]-V[1])/R3)-((V[1]-0.7)/R4)),1/L*(V[0]-Vec(V[2])),(1/L)*V[0]]


#def neuronode(V,t,L,E,C1,C2,C3,R1,R3,R4,I):
 #   return [(1/C1)*((E-V[0])/R1 - V[3] - V[4] - (V[0]-V[2])/R3 + I) , (1/C2)*V[4] , (1/C3)* ( (V[0]-V[2])/R3 - (V[2]-0.7)/R4), (1/L)*( V[0] - Vec(V[3]) ), (1/L)*V[1]]#(1/L)*(V[1]-Vce(V[4],(V[2]-0.7)/R3))]
"""
def neuronode(V,t,L,E,C1,C2,C3,R1,R3,R4,I):
    if (V[2]<=0.65):
        V[4]=0
        dI2=0
    else:
        dI2=(1 / L) * V[1]

    return [(1/C1)*((E-V[0])/R1 - V[3] - V[4] - (V[0]-V[2])/R3 + I) , (1/C2)*V[4], (1/C3)* ((V[0]-V[2])/R3), (1/L)*(V[0] - Vec(V[3])), dI2]#(1/L)*(V[1]-Vce(V[4],(V[2]-0.7)/R3))]
"""

def neuronode(V,t,L,E,C1,C2,C3,R1,R3,R4,I):

    if ((V[2])<0.6):
        Vout=[(1/C1)*((E-V[0])/R1 - V[3] - (V[0]-V[2])/R3 + I),0, (1/C3) * ( (V[0]-V[2])/R3), (1/L)*( V[0] - Vec(V[3]) ), 0]
    elif (V[2]>0.6):
        Vout=[(1 / C1)*((E - V[0]) / R1 - V[3] - (V[0] - V[2]) / R3 + I), (1/C2) * V[4], (1 / C3) * ((V[0] - V[2]) / R3 - (V[2]-0.7)/R4),
         (1 / L) * (V[0] - Vec(V[3])), V[0] - V[1] - Vce(V[4],(V[2]-0.7)/R4)]

    #Vout=[(1/C1)*((E-V[0])/R1 - V[3] - (V[0]-V[2])/R3 + I),0, (1/C3) * ( (V[0]-V[2])/R3), (1/L)*( V[0] - Vec(V[3]) ), 0]
    return Vout

"""
def neuronode(V,t,L,E,C1,C2,C3,R1,R3,R4,I):
    return [(1/C1)*((E-V[0])/R1 - V[3] - (V[0]-V[2])/R3 + I),0, (1/C3) * ( (V[0]-V[2])/R3), (1/L)*( V[0] - Vec(V[3]) ), 0]#(1/L)*(V[1]-Vce(V[4],(V[2]-0.7)/R3))]
"""

pvals=np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)

V0=[0,0,0,0,0]
T=100e-3
t=linspace(0,T,1000)

for i in range (0,10):
    L=1e-7
    R3=R3+i*20e3
    V=odeint(neuronode, V0, t, args=(L,E,C1,C2,C3,R1,R3,R4,I,))
    j = 0
    V1=[]
    V3=[]
    while (j <= 999):


        V1.append(V[j][0])
        V3.append(V[j][2])
        j = j + 1




print(V)
plot(t,V1,'b')
plot(t,V3,'r')
show()