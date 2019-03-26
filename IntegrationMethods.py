# -*- coding: utf-8 -*-
"""
This module contains the implementation of several methods of numerical 
integration
"""
import numpy as np

"""
The following 2 functions,
    Simpson
    adaptiveSimpson
are in conjunction with task 1 a) in the project.
"""

def Simpson(f,interval):
    """
    This is the Simpson rule
    Inputs:
        f = a function which integration shall be estimated
        Interval= a tuple with the start and ending point of the integration
    Output:
        The estimated integration of f over the interval
    """
    a,b=interval
    return ((b-a)/6)*(f(a)+4*f((a+b)/2)+f(b))
    
def adaptiveSimpson(f,interval,TOL):
    """
    This is the adaptive Simpson integration method
    Inputs:
        f = a function which integration shall be estimated
        Interval = a tuple with the start and ending point of the integration
        TOL = The bound we set upon the estimated error. 
    Output:
        The estimated definite integral of f
    
    The error estimate is here given by the absolute difference between the 
    simpson rule over the given interval and the composite simpson rule with
    two subintervals over the same interval, divided by 15. 15 comes from the
    fact that the composite simpson has 1/m**2 better error bound than normal
    simpson where m is the number of subintervals.
    """
    I1=Simpson(f,interval) #Simpsons Rule
    a,b=interval
    c=(a+b)/2
    I2=Simpson(f,(a,c))+Simpson(f,(c,b)) #Composite Simpson with two subint
    Error= (1/15)*np.abs(I2-I1)
    if Error<TOL:
        I2=I2+(1/15)*(I2-I1)
    else:
        I2=adaptiveSimpson(f,(a,c),TOL/2)+adaptiveSimpson(f,(c,b),TOL/2)
    return I2

"""
The following functions,
    compTrapezoid
    rombergIntegration
are in conjunction with task 1 c) of the project.
"""


def rombergIntegration(f,interval,m,TOL):
    """
    This is the Romberg integration method
    Inputs:
        f = a function which integration shall be estimated
        interval = a tuple with the start and ending point of the integration
        m = Maximum allowed dimension for the Romberg matrix
        TOL = The bound we set upon the estimated error. 
    Output:
        The estimated definite integral of f
    
    The method is implemented following the specifications in the task. 
    """
    #Initializing
    #Variables
    a,b=interval
    hn=(b-a)
    RombMatr=np.zeros((m,m))
    RombMatr[0][0]=(1/2)*hn*(f(a)+f(b))
    #Needed function
    def errCor(n,k):
        return (1/(4**(k)-1))*(RombMatr[n][k-1]-RombMatr[n-1,k-1])
    #Main
    for n in range(1,m):
        addition=0
        hn=hn*1/2
        for i in range(2**(n-1)):
            addition+=f(a+(2*i+1)*hn)
        RombMatr[n][0]=(1/2)*RombMatr[n-1][0]+hn*addition
        for k in range(1,n):
            RombMatr[n][k]=RombMatr[n][k-1]+errCor(n,k)
        if n !=1: #Else the code will divide by zero
            if np.abs(errCor(n,n-1))<TOL:
                return RombMatr[n][n-1]
    return RombMatr[-1][-1]