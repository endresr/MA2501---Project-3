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

# Plot
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

fig = plt.figure(1, figsize=(16, 9), dpi=100, facecolor='xkcd:pale',
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

print('-' * 50)
print("{:.^50}".format("Definite integral"))
print('-' * 50)
print("{:<20}{:>15}{:>15}".format(" ", "f(x)", "g(x)"))
print("{:<20}{:>15}{:>15}".format("Exact Integral", str(0), str(3 / 4)))
print("{:<20}{:>15}{:>15}".format("Adaptive Simpson", str(round(FaS, 6)),
                                  str(round(HaS, 6))))
print("{:<20}{:>15}{:>15}".format("Romberg Integration", str(round(Fr, 6)),
                                  str(round(Hr, 6))))

# Convergence
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


#Jalla = IntM.impMidRungKut((t0, tn), m0, funk, h, Jac)
#print(Jalla[:, -1])

#Reference = spi.solve_ivp(funk, (t0, tn), m0.reshape(1, 3)[0]).y[:, 2].reshape((3, 1))
#print(Reference)


# Jalla3=IntM.modiEul((t0,tn), m0, funk,h)
# print(Jalla3[-1])


# Jalla4=IntM.imprEul((t0,tn),m0,funk,h)
# print(Jalla4[-1])
with open('heavyCalc.pkl','rb') as d:
    dictValues=pickle.load(d)

oppg2c=dictValues.get('2c')
oppg2d=dictValues.get('2d')

#2c
fig=plt.figure(3)
plt.subplot(111)
plt.loglog(oppg2c[0],oppg2c[1],'x')
plt.loglog(oppg2c[0],oppg2c[2],'o')
plt.loglog(oppg2c[0],[x**2 for x in oppg2c[0]])
plt.show()

# gamma = IntM.gamma(Jalla3)
# print(gamma)
def gamSphere(m):
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    r = m.T @ m

    x = r * np.outer(np.cos(theta), np.sin(phi))
    y = r * np.outer(np.sin(theta), np.sin(phi))
    z = r * np.outer(np.ones(np.size(theta)), np.cos(phi))
    return x, y, z


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
    ax.plot(X, Y, Z, color="xkcd:crimson")
    ax.scatter(X[0], Y[0], Z[0], 'o', color="black", s=80)
    ax.view_init(azim=225)
    plt.title(Title)
    plt.show()


x, y, z = gamSphere(m0)
plotSphere(x, y, z, Jalla[0, :], Jalla[1, :], Jalla[2, :])

# gammaRef=
# EnerRef=

# gamma2 = IntM.gamma(Jalla4)
# print(gamma2)
