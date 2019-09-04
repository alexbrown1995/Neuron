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

v1a,v2,t1=readin('v1v23spikeburst.csv')
v1b,v3,t2=readin('v1v33spikeburst.csv')
v1c,v4,t=readin('v1vc2lowoscthing.csv')
vc2=array(v1c)-array(v4)
plot(t2,v1b)
plot(t2,v3)
show()

plot(t,v1c)
plot(t,vc2)
show()


plot(t1,v1a,'b')
plot(t,v1b,'r')
plot(t,v1c,'g')
show()

vc2=array(v1c)-array(v4)

t=array(t)-array(t[-1])


v1a=10*array(v1a)
v2=10*array(v2)
v3=10*array(v3)
vc2=10*array(vc2)
v1c=10*array(v1c)


v1a=v1a[::-1]
v1b=v1b[::-1]
v1c=v1c[::-1]
v3=v3[::-1]
vc2=vc2[::-1]
v2=v2[::-1]
t=t[::-1]


fig, ax1 = subplots()

color = 'tab:blue'
ax1.set_xlabel('time in ms',fontsize=10)
ax1.set_ylabel('Voltage at V1 and VC2', color=color,fontsize=10)
lns1=ax1.plot(t,v1c,'b',label="Voltage measured at V1",linewidth=2)
lns2=ax1.plot(t,vc2,'g',label="Voltage VC2 measured across capacitor C2",linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Voltage at V3 and V2', color=color,fontsize=10)  # we already handled the x-label with ax1
lns3=ax2.plot(t,v3,'r',label="Voltage measured at V3",linewidth=2)
lns4=ax2.plot(t1,v2,'c',label="Voltage measured at V2",linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=0,fontsize=10)
title("Voltage readings at four points throughout circuit for resistances ",fontsize=12)
show()




xlabel('time in ms',fontsize=10)
ylabel('Voltage at V1, V2, V3 and VC2',fontsize=10)
plot(t,v1c,'b',label="Voltage measured at V1",linewidth=2)
plot(t,vc2,'g',label="Voltage VC2 measured across capacitor C2",linewidth=2)



plot(t,v3,'r',label="Voltage measured at V3",linewidth=2)
plot(t1,v2,'c',label="Voltage measured at V2",linewidth=2)
legend()
title("Voltage readings at four points throughout circuit for resistances $R3=164.3K\Omega$, $R3_a=390K\Omega$ and $R3_b=28.96K\Omega$",fontsize=12)
show()

#scatter(t,v1a)
scatter(t1,v2)
scatter(t,v1c)

scatter(t,vc2)
show()


fig = figure()
ax = axes(projection="3d")


ax.scatter3D(v1c,vc2,v3,'r')
ax.set_xlabel('V1',fontsize=20)
ax.set_ylabel('V2',fontsize=20)
ax.set_zlabel('V3',fontsize=20)
title("3 dimensional phase plane diagram",fontsize=23)

show()
fig = figure()
ax = axes(projection="3d")


ax.plot(v1c,vc2,v3,'r')
ax.set_xlabel('V1',fontsize=20)
ax.set_ylabel('VC2',fontsize=20)
ax.set_zlabel('V3',fontsize=20)
title("3 dimensional phase plane diagram",fontsize=23)

show()

"""
v1,v2,t=readin('v1v2fasthighrez.csv')

v1=10*array(v1)
v2=10*array(v2)

plot(t,v1,label="V1")

plot(t,v2,label="V2")

xlabel("time in ms")

ylabel("voltage in V")
legend()
show()



plot(v1,v2)

xlabel("V1")

ylabel("V2")
show()
"""