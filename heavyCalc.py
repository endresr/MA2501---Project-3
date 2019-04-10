# -*- coding: utf-8 -*-
"""
Instead of running the codes several times, we run the needed ones here and 
save the results.
"""
import numpy as np
import scipy.integrate as spi

import IntegrationMethods as IntM 

import pickle

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
numbPoints=20

steps=10**np.linspace(-1,-4,numbPoints)
tend=1
RefEnd=spi.solve_ivp(funk,(t0,tend),m0.reshape(1,3)[0]).y[:,2].reshape((3,1))

Mid=np.array(
        [IntM.impMidRungKut((t0,tend),m0,funk,S,Jac)[:,-1].reshape(3,1) for S in steps]
        )
Eul=np.array(
        [IntM.modiEul((t0,tend), m0, funk,S)[:,-1].reshape(3,1) for S in steps]
        )

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

tend=50
h1=1e-1
h2=1e-2

Mid1=IntM.impMidRungKut((t0,tend),m0,funk,h1,Jac)
Mid2=IntM.impMidRungKut((t0,tend),m0,funk,h2,Jac)
Eul1=IntM.modiEul((t0,tend), m0, funk,h1) 
Eul2=IntM.modiEul((t0,tend), m0, funk,h2) 

print("Done with Mid1-Eul2")

tlist1=np.linspace(t0,tend,int((tend-t0)/h1))
tlist2=np.linspace(t0,tend,int((tend-t0)/h2))

Ref1=np.array(
        [spi.solve_ivp(funk,(t0,t),m0.reshape(1,3)[0]).y[:,-1].reshape((3,1))
        for t in tlist1])
    
print("Done with Ref1")

Ref2=np.array(
        [spi.solve_ivp(funk,(t0,t),m0.reshape(1,3)[0]).y[:,-1].reshape((3,1))
        for t in tlist2])
    
print("Done with Ref2")

def gam(m):
    return m.T @ m
def KinErg(m):
    return .5*m.T @ (Tinv @ m)

def Err2d(List,Ref):
    Worklist=List[:,1:]
    ArrErrGam=np.zeros(Worklist.shape[1])
    ArrErrKin=np.zeros(Worklist.shape[1])
    print(ArrErrKin.shape)
    for i in range(Worklist.shape[1]):
        ArrErrGam[i]=np.absolute(gam(Worklist[:,i])-gam(Ref[i].reshape(1,3)))
        ArrErrKin[i]=np.absolute(KinErg(Worklist[:,i])-KinErg(Ref[i].reshape(1,3)))
    print(ArrErrKin.shape)
    print(ArrErrGam.shape)
    return ArrErrGam,ArrErrKin

ErrMid1=Err2d(Mid1,Ref1)
ErrMid2=Err2d(Mid2,Ref2)
EulErr1=Err2d(Eul1,Ref1)
EulErr2=Err2d(Eul2,Ref2)

print("Done with Error oppg. 2d")

savingDict={'2c':[steps,ErrMid,EulErr],
            '2d':[tlist1,Mid1,Eul1,Ref1,ErrMid1,EulErr1,
                  tlist2,Mid2,Eul2,Ref2,ErrMid2,EulErr2]}

h = open("heavyCalc.pkl","wb")
pickle.dump(savingDict,h)
h.close()
print("Done! Saved to file")