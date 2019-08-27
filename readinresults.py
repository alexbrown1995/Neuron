from numpy import *
# Import the ODE integrator
from scipy.integrate import odeint
# Get access to the root-finder
from scipy.optimize import fsolve
# Plotting
from matplotlib.pyplot import *
import csv
from mpl_toolkits import mplot3d

with open('v1v21period.csv', 'r') as csvfile:
    voltages = csv.reader(csvfile, delimiter=',')
    print(voltages)
    v1a=[]
    v2=[]
    ta=[]
    linecount = 0
    for row in voltages:
        if linecount == 0:
            linecount += 1
        else:#if (float(row[0]) >= 0):

            v1a.append(float(row[1]))
            v2.append(float(row[2]))

            ta.append(float(row[0]))


with open('v1v31period.csv', 'r') as csvfile:
    voltages = csv.reader(csvfile, delimiter=',')
    print(voltages)
    v1b=[]
    v3=[]
    tb=[]
    linecount = 0
    for row in voltages:
        if linecount == 0:
            linecount += 1
        else:#if (float(row[0]) >= 0):

            v1b.append(float(row[1]))
            v3.append(float(row[2]))

            tb.append(float(row[0]))
plot(ta,v1a,'b')
plot(tb,v1b,'r')
plot(ta,v2)
plot(tb,v3)


show()
plot(v1a,v2)
show()
plot(v1b,v3)
plot(v2,v3)
show()

for i in range(len(ta)):
    if ta[i]==-38.184:
        print(i)
    if ta[i]==17.383:
        print(i)

for i in range(len(tb)):
    if tb[i]==-42.676:
        print(i)
    if tb[i]==12.793:
        print(i)


V1a=v1a[334:884]
V2=v2[334:884]
Ta=ta[334:884]

V1b=v1b[381:931]
V3=v3[381:931]
Tb=tb[381:931]

plot(Ta,V1a,'b')
plot(Ta,V1b,'r')
plot(Ta,V2)
plot(Ta,V3)


show()
plot(V1a,V2)
show()
plot(V1b,V3)
plot(V2,V3)
show()


fig = figure()
ax = axes(projection="3d")


ax.plot3D(V1a,V2,V3)
show()
# align data
"""
with open('v1andv2.csv', 'r') as csvfile:
    voltages = csv.reader(csvfile, delimiter=',')
    print(voltages)
    v1v2=[]
    v1=[]
    v2=[]
    t=[]

    linecount = 0
    for row in voltages:
        if linecount == 0:
            linecount += 1
        else:#if (float(row[0]) >= 0):
            v1v2.append([float(row[1]),float(row[2]),float(row[0])])
            v1.append(float(row[1]))
            v2.append(float(row[2]))
            t.append(float(row[0]))


with open('v1v3.csv', 'r') as csvfile:
    voltages = csv.reader(csvfile, delimiter=',')
    print(voltages)
    v1v3=[]
    linecount=0
    while linecount <= 494:
        for row in voltages:

            if linecount==0:
                linecount+=1
            else:
                v1v3.append([float(row[1]), float(row[2]), float(row[0])])
                linecount += 1
print(v1v2)
plot(t,v1)
show()
"""
"""
V1V2=[]
V1V3=[]
for i in range(len(v1v2)):
    for j in range(len(v1v3)):
        if v1v2[i][0]==v1v3[j][0]:
            #V1V3.append(v1v3[j])
            #V1V2.append(v1v3[i])
            print(i)
            print(v1v2[i])
            print(j)
            print(v1v3[j])
"""
"""
v1=v1[::-1]
v2=v2[::-1]
v3=v3[::-1]
print(len(v1))
print(len(v2))
print(len(v3))
fig = figure()
ax = axes(projection="3d")
print(v3)
x=[1,2,3]
print(x)
ax.plot3D(v3,v2,v1)
#plot(v1,v2)
show()
plot(v1,v2)
show()
plot(v1,v3)
show()
plot(v2,v3)
show()
plot(V1,V2)
show()
"""