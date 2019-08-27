from numpy import *
# Import the ODE integrator
from scipy.integrate import odeint
# Get access to the root-finder
from scipy.optimize import fsolve
# Plotting
from matplotlib.pyplot import *
import csv
from mpl_toolkits import mplot3d


with open('V1andVCE.csv', 'r') as csvfile:
    voltages = csv.reader(csvfile, delimiter=',')
    print(voltages)
    v1=[]
    vce=[]
    vals=[]
    t=[]
    linecount=0
    while linecount <= 508:
        for row in voltages:

            if linecount==0:
                linecount+=1
            else:
                if (float(row[0]) >= 0):
                    vals.append([float(row[0]),float(row[1]),float(row[0])])
                    #v1.append(float(row[1]))
                    #vce.append(float(row[2]))
                    #t.append(float(row[0]))
                linecount += 1

vals.sort(key=lambda x: x[1])
for val in vals:
    t.append(val[0])
    vce.append(val[2])
    v1.append(val[1])


plot(t,vce,'c')
plot(t,v1,'r')
show()

with open('V1andVBE.csv', 'r') as csvfile:
    voltages = csv.reader(csvfile, delimiter=',')
    print(voltages)
    V1=[]
    vbe=[]
    T=[]
    linecount=0
    while linecount <= 508:
        for row in voltages:

            if linecount==0:
                linecount+=1
            else:
                if (float(row[0])>=0):
                    V1.append(float(row[1]))
                    vbe.append(float(row[2]))
                    T.append(float(row[0]))
                linecount += 1
plot(T,vbe,'c')
plot(T,V1,'r')
show()