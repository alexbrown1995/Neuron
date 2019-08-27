from numpy import *
# Import the ODE integrator
from scipy.integrate import odeint
# Get access to the root-finder
from scipy.optimize import fsolve
# Plotting
from matplotlib.pyplot import *
import csv
from mpl_toolkits import mplot3d


def readin(name):
    with open(name, 'r') as csvfile:
        voltages = csv.reader(csvfile, delimiter=',')
        print(voltages)
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

v1a,v2,ta=readin('v1v2burstsorted.csv')
v1b,v3,tb=readin('v1v3burstsorted.csv')


v1g,v2g,t=readin('periodicorbitv1v2.csv')

scatter(v1a,ta)

scatter(v2,ta)
scatter(v3,ta)
scatter(v1b,ta)

show()
scatter(v1a,v2)
show()
scatter(v1b,v3)
show()
plot(ta,v1a,'b')
plot(ta,v1b,'r')
show()


fig = figure()
ax = axes(projection="3d")


ax.scatter3D(v1a,v2,v3)
show()

plot(v1g,v2g)
show()
"""
with open('v1v2fine.csv', 'r') as csvfile:
    voltages = csv.reader(csvfile, delimiter=',')
    print(voltages)
    v1=[]
    v2=[]
    t=[]
    linecount = 0
    for row in voltages:
        if linecount == 0:
            linecount += 1
        else:#if (float(row[0]) >= 0):

            v1.append(float(row[1]))
            v2.append(float(row[2]))
            t.append(float(row[0]))

scatter(v1,v2)

show()
"""