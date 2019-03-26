# -*- coding: utf-8 -*-
"""
This is the main module for project 3 in MA2501
Authors: Anne Bakkebø, Thomas Schjem and Endre Rundsveen
"""
import numpy as np
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

fig=plt.figure(1,figsize=(16,4),dpi=300,facecolor='xkcd:pale',
               edgecolor='none')
ax=fig.add_subplot(111)
ax.set_axis_bgcolor('xkcd:pale grey')
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