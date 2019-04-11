# -*- coding: utf-8 -*-
"""
Instead of calculationg large data sets for problem 2 several time, we 
collected all the computation in this module, and save the results in 
"heavyCalc.pkl". 
"""
import numpy as np
import scipy.integrate as spi #Needed for the reference ODE-solver
import IntegrationMethods as IntM #The methods used for integration
import pickle #Needed to save the results

t0=0
m0=np.array([[1],
             [1],
             [1]])
m0=m0/np.linalg.norm(m0)
L=np.array([1,2,3])
Tinv=np.diag(1/L)

def funk(t,m):
    return np.cross(m, Tinv @ m,axis=0)

def Jac(t, m):
    x1, x2, x3 = m[0,0],m[1,0],m[2,0]
    l1,l2,l3 = L
    J = np.array([[0,x3*(1/l3-1/l2),x2*(1/l3-1/l2)],
                      [x3*(1/l1-1/l3),0,x1*(1/l1-1/l3)],
                      [x2*(1/l2-1/l1),x1*(1/l2-1/l1),0]])
    return J

#Arrays for 2c

numbPoints=10

steps=10**np.linspace(-1,-4,numbPoints)
tend=1
RefEnd=spi.solve_ivp(funk,(t0,tend),
                     m0.reshape(1,3)[0],
                     rtol=1e-12,atol=1e-14).y[:,-1].copy().reshape((3,1))

Mid=np.array(
        [IntM.impMidRungKut((t0,tend),m0,funk,S,Jac)[:,-1].reshape(3,1) 
        for S in steps])
Eul=np.array(
        [IntM.imprEul((t0,tend), m0, funk,S)[:,-1].reshape(3,1) 
        for S in steps])

def Err(List):
    ArrErr=np.zeros(len(List))
    for i in range(len(ArrErr)):
        ArrErr[i]=np.linalg.norm(List[i]-RefEnd)
    return ArrErr

ErrMid=Err(Mid)
print("Done with ErrMid")
EulErr=Err(Eul)
print("Done with EulErr")

#Arrays for 2d and e

tend=30
h1=1e-1
h2=1e-2

Mid1=IntM.impMidRungKut((t0,tend),m0,funk,h1,Jac)
Mid2=IntM.impMidRungKut((t0,tend),m0,funk,h2,Jac)
Eul1=IntM.imprEul((t0,tend), m0, funk,h1) 
Eul2=IntM.imprEul((t0,tend), m0, funk,h2) 

print("Done with Mid1 - Eul2")

tlist1=np.linspace(t0,tend,int((tend-t0)/h1))
tlist2=np.linspace(t0,tend,int((tend-t0)/h2))
Ref1=spi.solve_ivp(funk,(t0,t0),m0.reshape(1,3)[0]).y[:,-1].reshape((3,1))
for t in tlist1:
    Ref1=np.append(Ref1,spi.solve_ivp(funk,
                                    (t0,t),
                                    m0.reshape(1,3)[0]).y[:,-1].reshape((3,1)),axis=1)

    
print("Done with Ref1")

Ref2=spi.solve_ivp(funk,(t0,t0),m0.reshape(1,3)[0]).y[:,-1].reshape((3,1))
for t in tlist2:
    Ref2=np.append(Ref2,spi.solve_ivp(funk,
                                    (t0,t),
                                    m0.reshape(1,3)[0]).y[:,-1].reshape((3,1)),axis=1)
    
print("Done with Ref2")

def gam(m):
    return np.linalg.norm(m)
def KinErg(m):
    return .5*np.inner(m.reshape(1,3),(Tinv @ m).reshape(1,3))

def Err2d(List):
    Worklist=List
    ArrErrGam=np.zeros(Worklist.shape[1])
    ArrErrKin=np.zeros(Worklist.shape[1])
    for i in range(Worklist.shape[1]):
        ArrErrGam[i]=np.absolute(
                gam(Worklist[:,i])-gam(m0))
        ArrErrKin[i]=np.absolute(
                KinErg(Worklist[:,i])-KinErg(m0))
    return ArrErrGam,ArrErrKin

ErrMid1=Err2d(Mid1)
ErrMid2=Err2d(Mid2)
EulErr1=Err2d(Eul1)
EulErr2=Err2d(Eul2)
ErrRef1=Err2d(Ref1)
ErrRef2=Err2d(Ref2)

print("Done with Error oppg. 2d")

savingDict={'2c':[steps,ErrMid,EulErr],
            '2d':[tlist1,Mid1,Eul1,Ref1,ErrMid1,EulErr1,ErrRef1,
                  tlist2,Mid2,Eul2,Ref2,ErrMid2,EulErr2,ErrRef2]}

h = open("heavyCalc.pkl","wb")
pickle.dump(savingDict,h)
h.close()
print("Done! Saved to file")