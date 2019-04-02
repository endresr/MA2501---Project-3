# -*- coding: utf-8 -*-
"""
This is the main module for project 3 in MA2501
Authors: Anne Bakkebø, Thomas Schjem and Endre Rundsveen
"""
import numpy as np
import sympy as sp
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import IntegrationMethods as IntM #The module containing our functions

from Tests import * #The tests are ran
"""
Task 1 
a) Is implemented in IntegrationMethods.py as adaptiveSimpson()
b) The test function for the adaptive Simpson Quadrature is ran in Tests.py
   The plots of the errors are given below
"""
#Variables and function
NumbPointa=100
TolRangea=np.linspace(10,1e-10,NumbPointa)
f=lambda x: np.cos(2*np.pi*x)
g=lambda x:np.exp(3*x)*np.sin(2*x)

FSim=np.asarray([IntM.adaptiveSimpson(f,(0,1),i) for i in TolRangea])
GSim=np.asarray([IntM.adaptiveSimpson(g,(0,np.pi/4),i) for i in TolRangea])
gint=(1/13)*(2+3*np.exp((3*np.pi)/4)) #Value of definite integral of g

#Plot

fig=plt.figure(1,figsize=(16,9),dpi=100,facecolor='xkcd:pale',
               edgecolor='none')
ax=fig.add_subplot(111)
ax.set_facecolor('xkcd:pale grey')
plt.xlim(TolRangea[0],TolRangea[-1])
plt.loglog(TolRangea,np.absolute(0-FSim),'xkcd:crimson',
           label=r'$f(x)=\cos(2\pi x),$ $x\in[0,1]$')
plt.loglog(TolRangea,np.absolute(gint-GSim),'k',
           label=r'$f(x)=e^{3x}\sin(2x)$ $x\in[0,\pi/4]$')
plt.loglog(TolRangea,TolRangea,'k--',label=r"Tolerance")
plt.xlabel("Tolerance value")
plt.ylabel(r"Error $\left(\int_a^bf(x)dx-\tilde{I}_{(a,b)}\right)$")
plt.title("Error of adaptive Simpson Rule")
plt.legend()
plt.show()

"""
c) Is implemented in IntegrationMethods.py as rombergIntegration
d) The test function for the Romberg integration is ran in Tests.py
    The rest of the task follows.
"""
#Comparing values from adaptive Simpson and Romberg
#Variables and function
h=lambda x:x**(1/3)
#Integrals
FaS=IntM.adaptiveSimpson(f,(0,1),1e-7)
Fr=IntM.rombergIntegration(f,(0,1),10,1e-7)
HaS=IntM.adaptiveSimpson(h,(0,1),1e-7)
Hr=IntM.rombergIntegration(h,(0,1),10,1e-7)

print('-'*50)
print("{:.^50}".format("Definite integral"))
print('-'*50)
print("{:<20}{:>15}{:>15}".format(" ","f(x)","g(x)"))
print("{:<20}{:>15}{:>15}".format("Exact Integral",str(0),str(3/4)))
print("{:<20}{:>15}{:>15}".format("Adaptive Simpson",str(round(FaS,6)),
      str(round(HaS,6))))
print("{:<20}{:>15}{:>15}".format("Romberg Integration",str(round(Fr,6)),
      str(round(Hr,6))))

"""
Task 2
a) and b) is implemented in IntegrationMethods.py. The midpoint is implemented
as impMidRungKut. As at the time of writing code, the formula for improved 
Euler method in the project text looks more like the formula for modified 
Euler method, both are implemented. They are found under the names imprEul and
modiEul, respectively. 


c),d) and e) are given in the following.
The code is hardcoded for three variable vectorfunctions and needs the user
to find the inverse jacobian. Surely we could have made the function more 
general, but for little gain.
"""

L=(1,2,3)#Tensor values
t0=0#Start-time
tn=1#end-time
h=1e-5
m0=np.array([[1],
             [1],
             [1]])
def funk(t,m):
    T=np.diag(L)
    return np.cross(m, T @ m,axis=0)

def Jac(t, m):
    x1, x2, x3 = m
    l1,l2,l3 = L
    J = np.array([[0,x3*(l3-l2),x2*(l3-l2)],
                      [x3*(l1-l3),0,x1*(l1-l3)],
                      [x2*(l2-l1),x1*(l2-l1),0]])
    return J

Jalla=IntM.impMidRungKut((t0,tn), m0, funk, h, Jac)
print(Jalla)
