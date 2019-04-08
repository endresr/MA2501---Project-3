# -*- coding: utf-8 -*-
"""
This module contains the testfunctions for the project
"""
import numpy as np
import IntegrationMethods as IntM


def test_adaptiveSimpson():
    f=lambda x: np.cos(2*np.pi*x)
    g=lambda x: np.exp(3*x)*np.sin(2*x)
    gval=(1/13)*(2+3*np.exp((3*np.pi)/4))
    assert np.abs(0-IntM.adaptiveSimpson(f,(0,1),1e-7))<1e-7, \
    "adaptive Simpson Quadrature gives wrong value for f(x)=cos(2*pi*x) over\
    [0,1]"
    assert np.abs(gval-IntM.adaptiveSimpson(g,(0,np.pi/4),1e-7))<1e-7, \
    "adaptive Simpson Quadrature gives wrong value for g(x)=exp(3*x)*sin(2x)\
    over [0,pi/4]"
test_adaptiveSimpson()

def test_rombergIntegration():
    f=lambda x: np.cos(2*np.pi*x)
    eIntf=0
    eIntg=3/4
    assert np.abs(0-IntM.rombergIntegration(f,(0,1),20,1e-7))<1e-7, \
    "Romberg Integration gives wrong value for f(x)=cos(2*pi*x) over\
    [0,1]"
    

test_rombergIntegration()

