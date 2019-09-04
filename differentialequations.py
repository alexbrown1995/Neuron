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
R3=200e3
R4=10e3
I=0.5e-3
E=10

x=np.array([0,0,0,0,0.096153846e-3,0.144230769e-3,0.211538462e-3,0.326923077e-3,0.5e-3,1e-3,1.692307692e-3,2e-3,2.5e-3,2.788461538e-3,2.865384615e-3])
y=np.array([0,0,0,0,7,7.58,8,8.5,8.388888889,7.875,7.5,7.611111111,8.208333333,8.5,8.611111111])

def polynomialeval(x,y,xval,deg):
    pvals=np.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
    p=0
    for i in range(0,deg+1):
        p=p+pvals[i]*(xval**(deg-i))
    return p

def Vec(ln):
    v=polynomialeval(x,y,ln,3)

    return v

def readin(name):
    with open(name, 'r') as csvfile:
        voltages = csv.reader(csvfile, delimiter=',')
        x = []
        y = []
        t = []
        linecount = 0
        for row in voltages:
            if linecount == 0:
                linecount += 1
            else:  # if (float(row[0]) >= 0):
                x.append(float(row[1]))
                y.append(float(row[2]))
                t.append(float(row[0]))
    return x,y,t


v1,v3,t1=readin('v1v3sorted2.csv')
v4,v5,t2=readin('v1vc2sorted2.csv')


#v1=v1[::-1]
#v3=v3[::-1]
plot(t1,v3)
plot(t1,v1)
show()
#v1,v3,t1=readin('v1v33spikeburst.csv')
#v4,v5,t2=readin('v1vc23spikeburst.csv')

plot(t2,v4)
plot(t2,v5)
show()

print(v1)

vc2=array(v4)-array(v5)
vc2=vc2[::-1]
plot(t2,vc2)
plot(t2,v1)
show()

plot(t1,v1)
plot(t1,v4)
plot(t1,v3)
plot(t1,vc2)
show()



"""
print(len(v1))
print(len(v4))
print(t1[-1])
"""
t1=array(t1)-array(t1[-1])
t2=array(t2)-array(t2[-1])


v1=v1[::-1]
v3=v3[::-1]
v4=v4[::-1]
v5=v5[::-1]
t1=t1[::-1]
t2=t2[::-1]
"""
print(t1[0])
print(t2[0])
print(t1[-1])
print(t2[-1])
"""
#T1=linspace(0,t1[-1]e-3,len(t1))


#T2=linspace(0,t2[-1]e-3,len(t2))
"""
plot(T1,v1,'b')
plot(T1,v3,'r')
plot(T1,v4,'g')
plot(T1,vc2,'c')
show()
print(t1[1]-t1[2])
print(t2[1]-t2[2])

print(t1)
print(t2)
"""



V1=10*array(v1)#[0:313]
V3=10*array(v3)#[0:313]
V4=10*array(v4)#[0:313]
Vc2=10*array(vc2)#[0:313]
T1=array(t1)#[0:313]
T2=array(t2)#[0:313]

#print(T1)

fig, ax1 = subplots()

color = 'tab:blue'
ax1.set_xlabel('time in ms',fontsize=20)
ax1.set_ylabel('Voltage at V1 and VC2', color=color,fontsize=20)
lns1=ax1.plot(T1[0:600],V1[300:900],'b',label="Voltage measured at V1",linewidth=2)
lns2=ax1.plot(T1[0:600],Vc2[300:900],'g',label="Voltage VC2 measured across capacitor C2",linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Voltage at V3', color=color,fontsize=20)  # we already handled the x-label with ax1
lns3=ax2.plot(T1[0:600],V3[300:900],'r',label="Voltage measured at V3",linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=0,fontsize=20)
#title("Voltage readings at three points throughout circuit",fontsize=35)
show()

"""
f,(ax,ax2) = subplots(2,1,sharex=True)

# plot the same data on both axes
lns1=ax.plot(T1[0:600],V1[300:900],'b',label="Voltage measured at V1",linewidth=2)
lns2=ax.plot(T1[0:600],Vc2[300:900],'g',label="Voltage VC2 measured across capacitor C2",linewidth=2)
lns3=ax2.plot(T1[0:600],V3[300:900],'r',label="Voltage measured at V3",linewidth=2)
ax.set_ylim(5,8)
ax2.set_ylim(0.2,0.8)

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')
ax2.xaxis.tick_bottom()
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

#plot(T1[0:600],V1[300:900],'b',label="Voltage measured at V1",linewidth=2)
xlabel("Time in ms")
ylabel("Voltage in V")
#ylim((0,8))




#plot(T1[0:600],V3[300:900],'r',label="Voltage measured at V3",linewidth=2)
#plot(T1[0:600],Vc2[300:900],'g',label="Voltage VC2 measured across capacitor C2",linewidth=2)
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=0)

show()
"""



def differential(t,vdata,tdata):
    h=tdata[1]-tdata[0]

    tmin=(t%(182.776))
    j=int(tmin/h)
    #print(j)
    """
    #finite difference
    if j==962:
        diff = (vdata[0]-vdata[j])/ (h)
    else:
        diff = (vdata[j+1] - vdata[j]) /(h)
    """
    #central difference



    if (1<=j<=(len(tdata)-2)):
        diff=(vdata[j+1]-vdata[j-1])/(2*h)
    elif(j==0):
        diff=(vdata[j+1]-vdata[-1])/(2*h)
    else:
        diff=(vdata[0]-vdata[j-1])/(2*h)

    return diff
dVc2=[]
for t in T1:
    dVc2.append(differential(t,Vc2,T1))

dV3=[]
for t in T1:
    dV3.append(differential(t,V3,T1))


plot(T1,dVc2,'g',label="Differential of VC2 obtained by central difference")

plot(T1,Vc2,'b',label="VC2")

xlabel("time in ms")
ylabel("Voltage and Voltage differentiated wrt time")
legend()
show()

plot(T1,dV3,'g',label="Differential of V3 obtained by central difference")

plot(T1,V3,'b',label="V3")

xlabel("time in ms")
ylabel("Voltage and Voltage differentiated wrt time")
legend()
show()


"""
fig = figure()
ax = axes(projection="3d")


ax.plot3D(V1,Vc2,V3,'r')
ax.set_xlabel('V1',fontsize=20)
ax.set_ylabel('VC2',fontsize=20)
ax.set_zlabel('V3',fontsize=20)
title("3 dimensional phase plane diagram",fontsize=23)

show()
"""


"""
plot(Vc2,V1)
plot(dVc2,V1)
show()
plot(Vc2,V3)
plot(dVc2,V3)
show()
"""
data=[]

for i in range(0,len(V1)):
    data.append([V1[i],Vc2[i],V3[i]]) #,dVc2[i]])

data=array(data)



import numpy as np
from itertools import combinations
import scipy.linalg

f = dVc2#np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).transpose()

G = data

#A = np.concatenate((G, np.ones((G.shape[0],1))), axis=1)
#C, _, _, _ = scipy.linalg.lstsq(A, f)
# C will have now the coefficients for:
# f(x, y, z) = ax + by + cz + d

# quadratic eq.
dim = G.shape[1]
#print(dim)
A = np.concatenate((G**2, np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose(), G, np.ones((G.shape[0], 1))), axis=1)
C, _, _, _ = scipy.linalg.lstsq(A, f)
# C will have now the coefficients for:
# f(x, y, z) = ax**2 + by**2 + cz**2 + dxy+ exz + fyz + gx + hy + iz + j
#print(C)
# This can be used then:
def quadratic(a):
    dim = a.shape[0]
    A = np.concatenate((a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim-1)]), a, [1]))
    return np.sum(np.dot(A, C))


A1 = np.concatenate((G**3,G**2, np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose(), G, np.ones((G.shape[0], 1))), axis=1)
C1, _, _, _ = scipy.linalg.lstsq(A1, f)
def cubic(a):
    dim = a.shape[0]
    A = np.concatenate((a**3,a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim - 1)]), a, [1]))
    return np.sum(np.dot(A, C1))





A3 = np.concatenate((G**5,G**4,G**3,G**2, np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose(), G, np.ones((G.shape[0], 1))), axis=1)
C3, _, _, _ = scipy.linalg.lstsq(A3, f)
def fifth(a):
    dim = a.shape[0]
    A = np.concatenate((a**5,a**4,a**3,a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim - 1)]), a, [1]))
    return np.sum(np.dot(A, C3))

"""
for i in range(G.shape[0]):
    print(cubic(G[i,:]))
"""
def differentialbetter(a,G,f):
    dim = a.shape[0]
    A2 = np.concatenate((G**3,G**2, np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose(), G, np.ones((G.shape[0], 1))), axis=1)
    C2, _, _, _ = scipy.linalg.lstsq(A2, f)

    A = np.concatenate((a**3,a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim - 1)]), a, [1]))
    return np.sum(np.dot(A, C2))

def differentialbetter2(x,y,z,fitto):
    print(x)
    A=concatenate((x**3,y**3,z**3,x**2*y,x**2*z,x*y**2,x*z**2,x*y*z,y**2*z,y*z**2,x*y,x*z,y*z,x,y,z,ones(len(x))))
    A=np.array([x**3,y**3,z**3,x**2*y,x**2*z,x*y**2,x*z**2,x*y*z,y**2*z,y*z**2,x*y,x*z,y*z,x,y,z,ones(len(x))])
    A=transpose(A)
    print(shape(A))
    print(shape(fitto))
    #print(A)
    C, _, _, _ = scipy.linalg.lstsq(A, fitto)#
    out=[]
    for i in range(0,len(x)):
        A1=np.array([x[i]**3,y[i]**3,z[i]**3,x[i]**2*y[i],x[i]**2*z[i],x[i]*y[i]**2,x[i]*z[i]**2,x[i]*y[i]*z[i],y[i]**2*z[i],y[i]*z[i]**2,x[i]*y[i],x[i]*z[i],y[i]*z[i],x[i],y[i],z[i],1])
        A1=transpose(A1)
        out.append(np.sum(np.dot(A1, C)))
    return out

def differentialbetter3(x1,y1,z1,x,y,z,fitto):
    A=concatenate((x**3,y**3,z**3,x**2*y,x**2*z,x*y**2,x*z**2,x*y*z,y**2*z,y*z**2,x*y,x*z,y*z,x,y,z,ones(len(x))))
    A=np.array([x**3,y**3,z**3,x**2*y,x**2*z,x*y**2,x*z**2,x*y*z,y**2*z,y*z**2,x*y,x*z,y*z,x,y,z,ones(len(x))])
    A=transpose(A)
    C, _, _, _ = scipy.linalg.lstsq(A, fitto)#
    A1=np.array([x1**3,y1**3,z1**3,x1**2*y1,x1**2*z1,x1*y1**2,x1*z1**2,x1*y1*z1,y1**2*z1,y1*z1**2,x1*y1,x1*z1,y1*z1,x1,y1,z1,1])
    A1=transpose(A1)
    return np.sum(np.dot(A1, C))


def differentialbetter(a,G,f):
    dim = a.shape[0]
    A2 = np.concatenate((G**10,G**9,G**8,G**7,G**6,G**5,G**4,G**3,G**2, np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose(), G, np.ones((G.shape[0], 1))), axis=1)
    C2, _, _, _ = scipy.linalg.lstsq(A2, f)

    A = np.concatenate((a**10,a**9,a**8,a**7,a**6,a**5,a**4,a**3,a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim - 1)]), a, [1]))
    return np.sum(np.dot(A, C2))
"""
def quartic(a):
    dim = a.shape[0]
    A = np.concatenate((a**4,a**3,a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim - 1)]), a, [1]))
    return np.sum(np.dot(A, C2))
"""

"""
for i in range(G.shape[0]):
    print(quadratic(G[i,:]), f[i])


for i in range(G.shape[0]):
    print(quadratic(G[i,:]), f[i])
"""
dataVc2=data
dataV3=data
#print(C)
"""
sol=[]
for val in data:
    sol.append(C[0]*val[0]**2+C[1]*val[1]**2+C[2]*val[2]**2+C[3]*val[0]*val[1]+C[4]*val[0]*val[2]+C[5]*val[1]*val[2]+C[6]*val[0]+C[7]*val[1]+C[8]*val[2]+C[9])


sol=[]
for i in range(G.shape[0]):
    sol.append(differentialbetter(G[i,:],dataVc2,dVc2))


sol=differentialbetter2(V1,Vc2,V3,dVc2)


fig = figure()
ax = fig.gca(projection='3d')
ax.scatter(sol,  data[:, 0], data[:,1], c='b', s=50,label="Cubic fitted solution")
ax.scatter(dVc2, data[:,0], data[:,1], c='r', s=50,label="Experimental data")
xlabel('Differential of VC2')
ylabel('V1')
ax.set_zlabel('VC2')
ax.legend()
show()

fig = figure()
ax = fig.gca(projection='3d')
ax.scatter(sol,  data[:, 1], data[:,2], c='b', s=50,label="Cubic fitted solution")
ax.scatter(dVc2, data[:,1], data[:,2], c='r', s=50,label="Experimental data")
xlabel('Differential of VC2')
ylabel('VC2')
ax.set_zlabel('V3')
ax.legend()
show()

fig = figure()
ax = fig.gca(projection='3d')
ax.scatter(sol,  data[:, 0], data[:,2], c='b', s=50,label="Cubic fitted solution")
ax.scatter(dVc2, data[:,0], data[:,2], c='r', s=50,label="Experimental data")
xlabel('Differential of VC2')
ylabel('V1')
ax.set_zlabel('V3')
ax.legend()
show()


sol2=[]
for i in range(G.shape[0]):
    sol2.append(differentialbetter(G[i,:],dataV3,dV3))


sol2=differentialbetter2(V1,Vc2,V3,dV3)
fig = figure()
ax = fig.gca(projection='3d')
ax.scatter(sol2,  data[:, 0], data[:,1], c='b', s=50,label="Cubic fitted solution")
ax.scatter(dV3, data[:,0], data[:,1], c='r', s=50,label="Experimental data")
xlabel('Differential of V3')
ylabel('V1')
ax.set_zlabel('VC2')
ax.legend()
show()

fig = figure()
ax = fig.gca(projection='3d')
ax.scatter(sol2,  data[:, 1], data[:,2], c='b', s=50,label="Cubic fitted solution")
ax.scatter(dV3, data[:,1], data[:,2], c='r', s=50,label="Experimental data")
xlabel('Differential of V3')
ylabel('VC2')
ax.set_zlabel('V3')
ax.legend()
show()

fig = figure()
ax = fig.gca(projection='3d')
ax.scatter(sol2,  data[:, 0], data[:,2], c='b', s=50,label="Cubic fitted solution")
ax.scatter(dV3, data[:,0], data[:,2], c='r', s=50,label="Experimental data")
xlabel('Differential of V3')
ylabel('V1')
ax.set_zlabel('V3')
ax.legend()
show()
"""
"""
 # regular grid covering the domain of the data
X,Y = np.meshgrid(np.arange(5, 8, 0.5),np.arange(5, 8, 0.5) )#np.arange(0.4, 0.8, 0.05))
XX = X.flatten()
YY = Y.flatten()

print(X)

# best-fit quadratic curve
A = c_[ones(data.shape[0]), data[:, :2], prod(data[:, :2], axis=1), data[:, :2] ** 2]
C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

# evaluate it on a grid
Z = dot(c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

# plot points and fitted surface
fig = figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
xlabel('X')
ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
show()

#print(T1)


def neuron(V,t,L):
    return[differential(t,V1,T1),differential(t,Vc2,T1),differential(t,V3,T1), (1/L)*(V[0] - Vec(V[3]) )]

def neuron2(V,t,L):
    return[differential(t,V1,T1),differentialbetter(array([V[0],V[1],V[2]]),dataVc2,dVc2),differentialbetter(array([V[0],V[1],V[2]]),dataV3,dV3), (1/L)*(V[0] - Vec(V[3]) )]


"""
T=200e-3
t=linspace(0,T,5000)
dvc2=[]

for i in range(len(V1)):
    dvc2.append(differentialbetter3(V1[i],Vc2[i],V3[i],V1,Vc2,V3,dVc2))


scatter(T1,dvc2)
scatter(T1,dVc2)
show()




dataVc2=data
dataV3=data


def neuron(V,t,L):
    return[differential(t,V1,T1),differential(t,Vc2,T1),differential(t,V3,T1), (1/L)*(V[0] - Vec(V[3]) )]

def neuron2(V,t,L):
    return[differential(t,V1,T1),differentialbetter3(V[0],V[1],V[2],V1,Vc2,V3,dVc2),differentialbetter3(V[0],V[1],V[2],V1,Vc2,V3,dVc2), (1/L)*(V[0] - Vec(V[3]))]

def neuronode(V,t,L,E,R1,R3,C2):
    return [ 1/C1*(((E-V[0])/R1)-V[3]-C2*differentialbetter3(V[0],V[1],V[2],V1,Vc2,V3,dVc2)-((V[0]-V[2])/R3)+I),C2*differentialbetter3(V[0],V[1],V[2],V1,Vc2,V3,dVc2),differentialbetter3(V[0],V[1],V[2],V1,Vc2,V3,dV3) ,(1/L)*(V[0] - Vec(V[3]))]





V0=[V1[0],Vc2[0],V3[0],0]
T=1000
t=linspace(0,T,10000)

L=1e-6
C1=1e-6
C2=10e-6
C3=2.2e-6
R1=3.3e3
R3=200e3
R4=10e3
I=0.6e-3
E=10
V = odeint(neuronode, V0, t, args=(L,E,R1,R3,C2))
j = 0
v1 = []
vc2=[]
v3 = []
In=[]
while (j <= 9999):
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

fig = figure()
ax = axes(projection="3d")


ax.scatter3D(v1,vc2,v3)
xlabel('V1')
ylabel('VC2')
ax.set_zlabel('V3')


show()

#print(V1)
#print(v1)
"""
plot(T1,sol,'b')
plot(T1,dVc2,'r')
show()

plot(T1,sol2,'b')
plot(T1,dV3,'r')
show()
val=[]



for i in range(0,1000):
    val.append(differential(i*0.19600000000002638*1e-3,V1,T1))

plot(t,val)
show()

h=(0.19600000000002638*1e-3)

print(62%60.73)

print(V1[20]-V1[18])

k=V1[20]-V1[18]
j=k/(1e-3)

print(j)
print(((V1[20]-V1[18])/h))


plot(v1,vc2)
show()
plot(v3,vc2)
show()
fig = figure()
ax = axes(projection="3d")


ax.plot3D(vc2,v1,v3)

show()

"""
#C=differentialbetter2(V1,Vc2,V3,dVc2)
#print(C)

print(Vec(0))