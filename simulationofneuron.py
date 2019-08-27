from numpy import *
# Import the ODE integrator
from scipy.integrate import odeint
# Get access to the root-finder
from scipy.optimize import fsolve
# Plotting
from matplotlib.pyplot import *
#from scipy.optimize import lmfit
#from lmfit import Model

L=1e-9
C1=1e-6
C3=2e-6
R1=3.3e3
R3=400e3
R4=1e3
E=10




def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


def line(x, slope, intercept):
    """a line"""
    return slope*x + intercept

#x= np.array([0, log(0.056818182e-3), log(0.102272727e-3), log(0.107954545e-3), log(0.15625e-3), log(0.3125e-3), log(0.5e-3), log(0.539772727e-3), log(0.625e-3), log(0.738636364e-3), log(0.965909091e-3),log(1.25e-3), log(1.647727273e-3), log(2.5e-3), log(5e-3), log(6.079545455e-3), log(6.704545455e-3)])
#y = np.array([7.25, 7.5, 8, 8.25, 8.45, 8.4626, 8.45, 8.25, 8.0625, 8, 7.84375, 7.71875, 7.68125, 7.75, 7.9125, 8, 8])
#x= np.log(np.array([0.02e-3, 0.056818182e-3, 0.102272727e-3, 0.107954545e-3, 0.15625e-3, 0.3125e-3, 0.5e-3, 0.539772727e-3, 0.625e-3, 0.738636364e-3, 0.965909091e-3,1.25e-3, 1.647727273e-3, 2.5e-3, 5e-3, 6.079545455e-3, 6.704545455e-3]))
#x3=np.array([0.02e-3, 0.056818182e-3, 0.102272727e-3, 0.107954545e-3, 0.15625e-3, 0.3125e-3, 0.5e-3, 0.539772727e-3, 0.625e-3, 0.738636364e-3, 0.965909091e-3,1.25e-3, 1.647727273e-3, 2.5e-3, 5e-3, 6.079545455e-3, 6.704545455e-3])
#x4=np.linspace(0.02e-3,7e-3,200)

#x=np.log(np.array([0.096153846e-3,0.144230769e-3,0.211538462e-3,0.326923077e-3,0.5e-3,1e-3,1.692307692e-3,2e-3,2.5e-3,2.788461538e-3,2.865384615e-3]))
#y=np.array([7,7.58,8,8.5,8.388888889,7.875,7.5,7.611111111,8.208333333,8.5,8.611111111])
#x3=np.array([0.096153846e-3,0.144230769e-3,0.211538462e-3,0.326923077e-3,0.5e-3,1e-3,1.692307692e-3,2e-3,2.5e-3,2.788461538e-3,2.865384615e-3])
#x4=np.linspace(0.096153846e-3,3e-3,200)

x=np.array([0.096153846e-3,0.144230769e-3,0.211538462e-3,0.326923077e-3,0.5e-3,1e-3,1.692307692e-3,2e-3,2.5e-3,2.788461538e-3,2.865384615e-3])
y=np.array([7,7.58,8,8.5,8.388888889,7.875,7.5,7.611111111,8.208333333,8.5,8.611111111])
x3=np.array([0.096153846e-3,0.144230769e-3,0.211538462e-3,0.326923077e-3,0.5e-3,1e-3,1.692307692e-3,2e-3,2.5e-3,2.788461538e-3,2.865384615e-3])
x4=np.linspace(0.096153846e-3,3e-3,200)


print(np.polyfit(x, y, 5, rcond=None, full=False, w=None, cov=True))
def polynomialeval(x,y,xval,deg):
    pvals=np.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
    p=0
    for i in range(0,deg+1):
        p=p+pvals[i]*(xval**(deg-i))
    return p
yval=[]
x2=np.log(np.linspace(0,7e-3,200))
for xval in x4:
    yval.append(polynomialeval(x,y,xval,5))
#print(xval)
#print(yval)
plot(x4,yval,"r-")
plot(x3,y,'bo')
show()
#x = linspace(0, 6, 17)
#mod = Model(gaussian) + Model(line)
#pars = mod.make_params(amp=1, cen=0.5, wid=1, slope=0, intercept=1)
#result = mod.fit(y_given, pars, x=x)

#plot(x_given, y_given, 'bo')
#plot(x_given, result.init_fit, 'k--')
#plot(x_given, result.best_fit, 'r-')
#show()
"""
def Vec(ln):

    predicted = mod.eval(result.params, x=ln)

    return predicted
"""

def Vec(ln):

    v=polynomialeval(x,y,np.log(ln),3)

    return v
def neuronode(V,t,pars,Iapp):
    return [(1 / pars[0])*(V[1]-Vec(V[0])+Iapp),(1/pars[1])*(((pars[3]-V[1])/pars[2])-V[0])]

def neuronodehigher(V,t,L,E,C1,C3,R1,R3,R4):
    return [(1/C1)*(((E-V[0])/R1)-V[2]-V[3]-((V[0]-V[1])/R3)),(1/C3)*(((V[0]-V[1])/R3)-((V[1]-0.7)/R4)),1/L*V[0]-Vec(V[2]),(1/L)*V[0]]

def completeneuronode(V,t,pars):
    return 1

#def Vec(ln):
  #  return -6.139*ln**3 + 4.5759*ln**2 - 0.069*ln + 0.0678

#def Vec(ln):
 #   return 4.64*ln**2 +0.4657*ln + 0.0626

#def Vec(ln):
 #   return 0.6877*ln + 0.0569

V0=[0,0]
V0H=[0,0,0,0]
T=100e-3
t=linspace(0,T,1000)
for i in range (0,10):
    L=1e-1+i*0.1e-1
    pars=[L,C1,R1,E]
    parsH=[L,E,C1,C3,R1,R3,R4]
    V = odeint(neuronode, V0, t, args=(pars,0.59e-3,))
    VH=odeint(neuronodehigher,V0H,t,args=(L,E,C1,C3,R1,R3,R4,))
    V1 = []
    V1H = []
    ln = []
    # print(len(V))
    j = 0
    while (j <= 999):
        ln.append(V[j][0])
        V1.append(V[j][1])
        V1H.append(V[j][0])
        j = j + 1

    print(ln)
    #plot(ln, V1)
    #show()
    plot(t, V1)

    show()


print(V)



"""
x_given=np.array([0,0.056818182,0.102272727,0.107954545,0.15625,0.3125,0.5,0.539772727,0.625,0.738636364,0.965909091,1.25,1.647727273,2.5,5,6.079545455,6.704545455])
y_given=np.array([7.25,7.5,8,8.25,8.45,8.4626,8.45,8.25,8.0625,8,7.84375,7.71875,7.68125,7.75,7.9125,8,8])
x=linspace(0,6,17)
"""


"""
i=0
while i<=5:


    mod = Model(gaussian) + Model(line)
    pars = mod.make_params(amp=1, cen=0.5, wid=1, slope=0, intercept=1)
    result = mod.fit(y_given, pars, x=x)
    y_given=np.append(y_given,result.best_fit)
    np.sort(y_given, axis=-1, kind='quicksort', order=None)
    x=np.linspace(0,6,len(y_given))
    i=i+1
"""


"""
mod = Model(gaussian) + Model(line)
pars = mod.make_params(amp=1, cen=0.5, wid=1, slope=0, intercept=1)

result = mod.fit(y_given, pars, x=x)

print(result.fit_report())
print(x)
print(result.best_fit)
plot(x_given, y_given, 'bo')
#plot(x_given, result.init_fit, 'k--')
plot(x_given, result.best_fit, 'r-')
show()
#comps = result.eval_components()
#plot(x_given, y_given, 'bo')
#plot(x_given, comps['gaussian'], 'k--')
#plot(x_given, comps['line'], 'r--')
#show()
x=np.linspace(0,5,100)
#predicted = mod.eval(pars,x=x)
#plot(x,predicted)
#show()
predicted = mod.eval(result.params,x=5)
print(predicted)
#print(result.params)
#plot(x, y_given, 'bo')
#plot(x,predicted)
#show()
"""
"""
x_p=linspace(0,7,100)
p3=polyfit(x_given,y_given,4)

y_p=polyval(p3,x_p)
plot(x_given,y_given,'o')
plot(x_p,y_p,'-')
show()
"""