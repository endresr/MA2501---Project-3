# -*- coding: utf-8 -*-
"""
This is the main module for project 3 in MA2501
Authors: Anna Bakkebø, Thomas Schjem and Endre Rundsveen
"""
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import IntegrationMethods as IntM  # The module containing our functions

from Tests import *  # The tests are ran

"""
Task 1 
a) Is implemented in IntegrationMethods.py as adaptiveSimpson()
b) The test function for the adaptive Simpson Quadrature is ran in Tests.py
   The plots of the errors are given below
"""

# Variables and function

NumbPointa = 100
TolRangea = np.linspace(1, 1e-11, NumbPointa)
f = lambda x: np.cos(2 * np.pi * x)
g = lambda x: np.exp(3 * x) * np.sin(2 * x)
FaS = IntM.adaptiveSimpson(f, (0, 1), 1e-7)
GaS = IntM.adaptiveSimpson(g, (0, np.pi / 4), 1e-7)

FSim = np.asarray([IntM.adaptiveSimpson(f, (0, 1), i) for i in TolRangea])
GSim = np.asarray([IntM.adaptiveSimpson(g, (0, np.pi / 4), i) for i in TolRangea])
gint = (1 / 13) * (2 + 3 * np.exp((3 * np.pi) / 4))  # Value of definite integral of g


#Table of values
print('-' * 60)
print("{:.^60}".format("Definite integral"))
print('-' * 60)
print("{:<20}{:>20}{:>20}".format(" ", "f(x)", "g(x)"))
print("{:<20}{:>20}{:>20}".format("Exact Integral", str(0),
                                  r"1/13(2+3exp(3pi/4)"))
print("{:<20}{:>20}{:>20}".format("Adaptive Simpson", str(round(FaS, 14)),
                                  str(round(GaS, 14))))
print("{:<20}{:>20}{:>20}".format("Error", str(np.absolute(round(FaS - 0, 14))),
                                  str(round(np.absolute(GaS - gint), 14))))

# Plot
fig = plt.figure(1, figsize=(16, 9), facecolor='xkcd:pale',
                 edgecolor='none')
ax = fig.add_subplot(111)
ax.set_facecolor('xkcd:pale grey')
plt.xlim(TolRangea[0], TolRangea[-1])
plt.loglog(TolRangea, np.absolute(0 - FSim), 'xkcd:crimson',
           label=r'$f(x)=\cos(2\pi x),$ $x\in[0,1]$')
plt.loglog(TolRangea, np.absolute(gint - GSim), 'k',
           label=r'$f(x)=e^{3x}\sin(2x)$ $x\in[0,\pi/4]$')
plt.loglog(TolRangea, TolRangea, 'k--', label=r"Tolerance")
plt.xlabel("Tolerance value")
plt.ylabel(r"Error, $\left|\int_a^bf(x)dx-\tilde{I}_{(a,b)}\right|$")
plt.title("Error of adaptive Simpson Rule")
plt.legend()
plt.show()

"""
c) Is implemented in IntegrationMethods.py as rombergIntegration
d) The test function for the Romberg integration is ran in Tests.py
    The rest of the task follows.
"""

# Comparing values from adaptive Simpson and Romberg
# Variables and function

h = lambda x: x ** (1 / 3)
# Integrals

Fr = IntM.rombergIntegration(f, (0, 1), 10, 1e-7)
HaS = IntM.adaptiveSimpson(h, (0, 1), 1e-7)
Hr = IntM.rombergIntegration(h, (0, 1), 10, 1e-7)

#Table of values
print('-' * 50)
print("{:.^50}".format("Definite integral"))
print('-' * 50)
print("{:<20}{:>15}{:>15}".format(" ", "f(x)", "g(x)"))
print("{:<20}{:>15}{:>15}".format("Exact Integral", str(0), str(3 / 4)))
print("{:<20}{:>15}{:>15}".format("Adaptive Simpson", str(round(FaS, 7)),
                                  str(round(HaS, 7))))
print("{:<20}{:>15}{:>15}".format("Romberg Integration", str(round(Fr, 7)),
                                  str(round(Hr, 7))))

# Convergence plot

IterN = 20  # Max iterations for convergence test
Converg1 = IntM.rombergIntegration(f, (0, 1), IterN, 1e-10, Matr=True)
Converg2 = IntM.rombergIntegration(h, (0, 1), IterN, 1e-10, Matr=True)

fig = plt.figure(2, figsize=(16, 4), dpi=100, facecolor='xkcd:pale',
                 edgecolor='none')
plt.subplot(121)
plt.semilogy(range(IterN), np.absolute(Converg1[:, 0] - 0), 'xkcd:crimson',
             label=r'$\varepsilon(n,0)=\left|\int_a^bf(x)dx-R(n,0)\right|$')
plt.semilogy(range(IterN), np.absolute(np.diagonal(Converg1) - 0), 'k',
             label=r'$\varepsilon(n,n)=\left|\int_a^bf(x)dx-R(n,n)\right|$')
plt.xlabel(r"Number of iterations $n$")
plt.ylabel(r"Error, $\left(\int_a^bf(x)dx-\tilde{I}_{(a,b)}\right)$")
plt.title(r"Error of Romberg for $f(x)=\cos(2\pi x),$ $x\in[0,1]$")
plt.legend()
plt.subplot(122)
plt.semilogy(range(IterN), np.absolute(Converg2[:, 0] - 0), 'xkcd:crimson',
             label=r'$\varepsilon(n,0)=\left|\int_a^bf(x)dx-R(n,0)\right|$')
plt.semilogy(range(IterN), np.absolute(np.diagonal(Converg2) - 0), 'k',
             label=r'$\varepsilon(n,n)=\left|\int_a^bf(x)dx-R(n,n)\right|$')
plt.xlabel(r"Number of iterations $n$")
plt.ylabel(r"Error, $\left(\int_a^bf(x)dx-\tilde{I}_{(a,b)}\right)$")
plt.title(r"Error of Romberg for $f(x)=x^{\frac{1}{3}},$ $x\in[0,1]$")
plt.legend()
plt.show()

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

L = (1, 2, 3)  # Tensor values
t0 = 0  # Start-time
tn = 1  # end-time
h = 1e-3
m0 = np.array([[1],
               [1],
               [1]])
m0 = m0 / np.linalg.norm(m0)
Tinv = np.diag((1 / L[0], 1 / L[1], 1 / L[2]))


def funk(t, m):
    return np.cross(m, Tinv @ m, axis=0)


def Jac(t, m):
    x1, x2, x3 = m[0, 0], m[1, 0], m[2, 0]
    l1, l2, l3 = L
    J = np.array([[0, x3 * (1 / l3 - 1 / l2), x2 * (1 / l3 - 1 / l2)],
                  [x3 * (1 / l1 - 1 / l3), 0, x1 * (1 / l1 - 1 / l3)],
                  [x2 * (1 / l2 - 1 / l1), x1 * (1 / l2 - 1 / l1), 0]])
    return J


#Plots of task 2c,d,e
with open('heavyCalc.pkl','rb') as d:
    dictValues=pickle.load(d)

oppg2c=dictValues.get('2c') 
'''The calculations needed for 2c are stored in this list.
oppg2c[0]=array of stepsizes.
oppg2c[1]=array of errors for the midpoint method over the interval [0,1] 
    corresponding to the step sizes in the previous array.
oppg2c[2]=array of errors for the improved euler method over the interval [0,1] 
    corresponding to the step sizes in the first array.
'''
oppg2d=dictValues.get('2d')
'''
oppg2d[0:6]=is the same as oppg2d[7:13] with step size 1e-1 instead of 1e-2
oppg2d[0]=Array of timepoints from 0 to 30 with step size 1e-1
oppg2d[1]=Array of the result from midpoint method over the interval [t0,tn] 
    where t0 is 0 and tn corresponds to the elements of oppg2d[0]
oppg2d[2]=Array of the result from improved euler method over the interval [t0,tn] 
    where t0 is 0 and tn corresponds to the elements of oppg2d[0]
oppg2d[3]=Array of the result from built in method over the interval [t0,tn] 
    where t0 is 0 and tn corresponds to the elements of oppg2d[0]
oppg2d[4][0]=Array of the error in distance from origo for positions from 
    midpoint method over the interval [0,tn], where tn comes from oppg2d[0]
oppg2d[4][1]=Array of the error in energy for positions from midpoint method 
    over the interval [0,tn], where tn comes from oppg2d[0]
oppg2d[5][0]=Array of the error in distance from origo for positions from 
    improved euler over the interval [0,tn], where tn comes from oppg2d[0]
oppg2d[5][1]=Array of the error in energy for positions from improved euler 
    over the interval [0,tn], where tn comes from oppg2d[0]
oppg2d[6][0]=Array of the error in distance from origo for positions from 
    built-in method over the interval [0,tn], where tn comes from oppg2d[0]
oppg2d[6][1]=Array of the error in energy for positions from built-in method 
    over the interval [0,tn], where tn comes from oppg2d[0]

oppg2d[14][0]+oppg2d[14][1] is the same as oppg2d[4][0]+oppg2d[4][1] only with 
    another tolerance in Newtons method
oppg2d[15][0]+oppg2d[15][1] is the same as oppg2d[5][0]+oppg2d[5][1] only with
    different relative and absolute tolerance
'''
#2c
fig=plt.figure(figsize=(16,4),facecolor="xkcd:pale")
plt.title("Errors of the methods for decreasing step size")
plt.loglog(oppg2c[0],oppg2c[2],
           'ks',markerfacecolor="none",
           label=r"Improved Euler")
plt.loglog(oppg2c[0],oppg2c[1],
           'x',color="xkcd:crimson",
           label=r"Midpoint RK")
plt.loglog(oppg2c[0],[x**2 for x in oppg2c[0]],
           'k--', label=r"$h^2$")
plt.legend(loc=4)
plt.ylabel(r"Error")
plt.xlabel(r"Step size, \(h\)")
plt.show()


#2d

fig,axes=plt.subplots(3,2,
    sharex=True,
    figsize=(10,10),
    facecolor='xkcd:pale')
fig.suptitle("Errors of the methods for increasing time")
axes[0,0].set_title("RK midpoint")
axes[0,0].semilogy(oppg2d[0],oppg2d[4][0],'x',
           color="xkcd:crimson",label=r"Error in $\gamma$")
axes[0,0].semilogy(oppg2d[0],oppg2d[4][1],'k--',
           label=r"Error in Energy $E$")
axes[0,0].set_ylabel(r"Error")
axes[0,0].legend(loc=4)

axes[1,0].set_title("Improved Euler")
axes[1,0].semilogy(oppg2d[0],oppg2d[5][0],'x',
           color="xkcd:crimson",label=r"Error in $\gamma$")
axes[1,0].semilogy(oppg2d[0],oppg2d[5][1],'k--',
           label=r"Error in Energy $E$")
axes[1,0].set_ylabel(r"Error")
axes[1,0].legend(loc=4)

axes[2,0].set_title("Built-in")
axes[2,0].semilogy(oppg2d[0],oppg2d[6][0],'x',
           color="xkcd:crimson",label=r"Error in $\gamma$")
axes[2,0].semilogy(oppg2d[0],oppg2d[6][1],'k--',
           label=r"Error in Energy $E$")
axes[2,0].set_ylabel(r"Error")
axes[2,0].legend(loc=4)
axes[2,0].set_xlabel("Time")

axes[0,1].set_title("RK midpoint")
axes[0,1].semilogy(oppg2d[7],oppg2d[11][0],'x',
           color="xkcd:crimson",label=r"Error in $\gamma$")
axes[0,1].semilogy(oppg2d[7],oppg2d[11][1],'k--',
           label=r"Error in Energy $E$")
#axes[0,1].set_ylabel(r"Error")
axes[0,1].legend(loc=4)

axes[1,1].set_title("Improved Euler")
axes[1,1].semilogy(oppg2d[7],oppg2d[12][0],'x',
           color="xkcd:crimson",label=r"Error in $\gamma$")
axes[1,1].semilogy(oppg2d[7],oppg2d[12][1],'k--',
           label=r"Error in Energy $E$")
#axes[1,1].set_ylabel(r"Error")
axes[1,1].legend(loc=4)

axes[2,1].set_title("Built-in")
axes[2,1].semilogy(oppg2d[7],oppg2d[13][0],'x',
           color="xkcd:crimson",label=r"Error in $\gamma$")
axes[2,1].semilogy(oppg2d[7],oppg2d[13][1],'k--',
           label=r"Error in Energy $E$")
#axes[2,1].set_ylabel(r"Error")
axes[2,1].legend(loc=4)
axes[2,1].set_xlabel("Time")

plt.show()

#Plot with lower tolerances
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
fig.suptitle("Errors of the methods for increasing time")
ax1.set_title("RK midpoint with tolerance 1e-10")
ax1.semilogy(oppg2d[0],oppg2d[14][0],'x',
           color="xkcd:crimson",label=r"Error in $\gamma$")
ax1.semilogy(oppg2d[0],oppg2d[14][1],'k--',
           label=r"Error in Energy $E$")
ax1.set_ylabel(r"Error")
ax1.legend(loc=4)
ax2.set_title("Built in with rtol=1e-10,atol=1e-12")
ax2.semilogy(oppg2d[0],oppg2d[15][0],'x',
           color="xkcd:crimson",label=r"Error in $\gamma$")
ax2.semilogy(oppg2d[0],oppg2d[15][1],'k--',
           label=r"Error in Energy $E$")
ax2.set_ylabel(r"Error")
ax2.legend(loc=4)

#2e

def gamSphere(m):
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    r = np.sqrt(m.T @ m)

    x = r * np.outer(np.cos(theta), np.sin(phi))
    y = r * np.outer(np.sin(theta), np.sin(phi))
    z = r * np.outer(np.ones(np.size(theta)), np.cos(phi))
    return x, y, z

'''
We've got both plotSphere and plotSphereDouble, where the only difference is
    that the last one plots for two different sets of points. Thus we can see
    the plots from two different step sizes, side by side.
'''
def plotSphere(x, y, z, X, Y, Z, Title="ODE-Solver"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    plt.axis('off')
    ax.plot_surface(x, y, z,
                    rstride=4,
                    cstride=4,
                    alpha=0.5,
                    color='xkcd:pale',
                    edgecolors="darkgray")
    ax.plot(X, Y, Z, 
            color="xkcd:crimson")
    ax.scatter(X[0], Y[0], Z[0], 
               'o', color="black", s=80)
    ax.view_init(azim=225)
    plt.title(Title)
    plt.show()

def plotSphereDouble(x,y,z,
                     X1,Y1,Z1,X2,Y2,Z2,
                     Title1="ODE-Solver",Title2="ODE-Solver",
                     MTitle="Two spheres"):
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    fig.suptitle(MTitle)
    ax1.axis("off"),ax2.axis("off")
    ax1.plot_surface(x, y, z,
                    rstride=4,
                    cstride=4,
                    alpha=0.5,
                    color='xkcd:pale',
                    edgecolors="darkgray")
    ax1.plot(X1, Y1, Z1, 
            color="xkcd:crimson")
    ax1.scatter(X1[0], Y1[0], Z1[0], 
               'o', color="black", s=80)
    ax1.view_init(azim=225)
    ax1.set_title(Title1)
    ax2.plot_surface(x, y, z,
                    rstride=4,
                    cstride=4,
                    alpha=0.5,
                    color='xkcd:pale',
                    edgecolors="darkgray")
    ax2.plot(X2, Y2, Z2, 
            color="xkcd:crimson")
    ax2.scatter(X2[0], Y2[0], Z2[0], 
               'o', color="black", s=80)
    ax2.view_init(azim=225)
    ax2.set_title(Title2)
    plt.show()
    
x, y, z = gamSphere(m0)
plotSphereDouble(x, y, z, 
           oppg2d[1][0, :], oppg2d[1][1, :], oppg2d[1][2, :],
           oppg2d[8][0, :], oppg2d[8][1, :], oppg2d[8][2, :],
           Title1="Step size 0.1",Title2="Step size 0.01",
           MTitle="Midpoint Runge Kutta")
plotSphereDouble(x, y, z, 
           oppg2d[2][0, :], oppg2d[2][1, :], oppg2d[2][2, :],
           oppg2d[9][0, :], oppg2d[9][1, :], oppg2d[9][2, :],
           Title1="Step size 0.1",Title2="Step size 0.01",
           MTitle="Improved Euler")
plotSphereDouble(x, y, z, 
           oppg2d[3][0, :], oppg2d[3][1, :], oppg2d[3][2, :],
           oppg2d[10][0, :], oppg2d[10][1, :], oppg2d[10][2, :],
           Title1="Step size 0.1",Title2="Step size 0.01",
           MTitle="Built in method")
