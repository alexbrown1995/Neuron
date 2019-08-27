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

        linecount = 0
        for row in voltages:
            if linecount == 0:
                linecount += 1
            else:  # if (float(row[0]) >= 0):

                x.append(float(row[0]))
                y.append(float(row[1]))

    return x,y

I,vbe=readin('trans_2n2222data.csv')

plot(vbe,I)

x=polyfit(vbe, log(I), 1, w=sqrt(I))

print(x)

v=linspace(0.05,0.65,100)

y=[]
for val in v:
    y.append(e**(x[1])*e**(x[0]*val))

plot(v,y)
show()