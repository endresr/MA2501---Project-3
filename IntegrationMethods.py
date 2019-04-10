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

def Simpson(f, interval):
    """
    This is the Simpson rule
    Inputs:
        f = a function which integration shall be estimated
        Interval= a tuple with the start and ending point of the integration
    Output:
        The estimated integration of f over the interval
    """
    a, b = interval
    return ((b-a)/6)*(f(a)+4*f((a+b)/2)+f(b))
    
def adaptiveSimpson(f, interval, TOL):
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
    I1 = Simpson(f, interval) #Simpsons Rule
    a, b = interval
    c = (a+b)/2
    I2 = Simpson(f, (a, c))+Simpson(f, (c, b)) #Composite Simpson with two subint
    Error = (1/15)*np.abs(I2-I1)
    if Error < TOL:
        I2 = I2+(1/15)*(I2-I1)
    else:
        I2 = adaptiveSimpson(f, (a, c), TOL/2)+adaptiveSimpson(f, (c, b), 
                             TOL/2)
    return I2

"""
The following functions,
    compTrapezoid
    rombergIntegration
are in conjunction with task 1 c) of the project.
"""


def rombergIntegration(f, interval, m, TOL,Matr=False):
    """
    This is the Romberg integration method
    Inputs:
        f = a function which integration shall be estimated
        interval = a tuple with the start and ending point of the integration
        m = Maximum allowed dimension for the Romberg matrix
        TOL = The bound we set upon the estimated error. 
        Matr = whether you want the whole matrix or not.
    Output:
        The estimated definite integral of f
    
    The method is implemented following the specifications in the task. 
    """
    #Initializing
    #Variables
    a, b = interval
    hn = (b-a)
    RombMatr = np.zeros((m, m))
    RombMatr[0, 0] = (1/2)*hn*(f(a)+f(b))
    #Needed function
    def errCor(n, k):
        return (1/(4**(k)-1))*(RombMatr[n, k-1]-RombMatr[n-1, k-1])
    #Main
    for n in range(1, m):
        addition = 0
        hn = hn*1/2
        for i in range(1,2**(n-1)+1):
            addition += f(a+(2*i-1)*hn)
        RombMatr[n][0] = (1/2)*RombMatr[n-1, 0]+hn*addition
        for k in range(1, n+1):
            RombMatr[n,k] = RombMatr[n, k-1]+errCor(n, k)
        if n != 1: 
            if np.abs(errCor(n, n-1)) < TOL:
                if Matr:
                    return RombMatr
                else:
                    return RombMatr[n, n-1]
    if Matr:
        return RombMatr
    else:
        return RombMatr[-1, -1]

"""
Implicit Midpoint Runge-Kutta 
"""

def Newt(func, Jac, x0, t, NIter=100, TOL=1e-7):
    """
    Newton method for finding the implicit functions in impMidRungKut
    Note that as its mother function, this function is hardcoded for
        three variables as well.
    Input:
        func: function which fixed is to be found for
        Jac: the jacobian of the function
        x0: Starting point
        t: the time which the functions fixed point is to be found
        NIter: Maximum iterations
        TOL: Tolerance for the answer
    Output:
        The fixed-point
    """
    i = 0
    x1 = x0
    while i < NIter:
        xalm=np.linalg.solve(Jac(t,x1),func(t,x1))
        x2 = x1-xalm
        Fx01 = func(t, x2)
        if np.linalg.norm(Fx01) < TOL or np.linalg.norm(x2-x1) < TOL:
            return x2
        x1 = x2
        i += 1
    return x2

def impMidRungKut(Interval, InitVal, F, Step, Jac):
    """
    This is the code for the implicit Midpoint Runge-Kutta method.
    Note that this is hardcoded for three variables, and uses Newton method
        to estimate the vectorfunctions. 
    Input:
        Interval: [t0,t] where a is start-time, and t is where the function
            shall be evaluated.
        InitVal: Value of function at time t0
        F: function for the derivative dependent on time and value 
            of function in question
        Step: Step-size of the method
        Jac: One has to provide the jacobian for the expression of the 
            derivative
    Output:
        The function at time t
    """
    a,b=Interval
    yn = InitVal
    tim = np.linspace(a,b,(b-a)/Step)
    for t in tim:
        JacK=lambda t,K: np.diag((1,1,1))-Jac(t+Step/2,yn[:,-1]+Step/2*K)
        Fu=lambda t,K: K-F(t+Step/2,yn[:,-1]+Step/2*K)
        K1 = Newt(Fu, JacK, np.zeros((3,1)), t)
        yn=np.append(yn,yn[:,-1]+Step*K1,axis=1)
    return yn

"""Per 28.03.18 18:40 kan det se ut som det er oppgitt formelen for modified
Euler der det står at vi skal implementere improved Euler. Har dermed begge 
i det følgende. Ref. Süli og Mayers (s. 328)"""

def modiEul(Interval, InitVal, F, Step):
    """
    This is the modified Euler method.
    Input:
        Interval: [t0,t] where a is start-time, and t is where the function
            shall be evaluated.
        InitVal: Value of function at time t0
        F: formula for the derivative dependent on time and value of function
        Step: Step-size of the method
    Output:
        The function at time t
    """
    a,b=Interval
    tim = np.linspace(a,b,(b-a)/Step)
    yn = InitVal
    for tn in tim:
        ynhalf = yn[:,-1]+.5*Step*F(tn, yn[:,-1])
        yn=np.append(yn,yn[:,-1]+Step*F(tn+.5*Step, ynhalf),axis=1)
    return yn

def imprEul(Interval, InitVal, F, Step):
    """
    This is the improved Euler method.
    Input:
        Interval: [t0,t] where a is start-time, and t is where the function
            shall be evaluated.
        InitVal: Value of function at time t0
        F: formula for the derivative dependent on time and value of function
        Step: Step-size of the method
    Output:
        The function at time t
    """
    a,b=Interval
    tim = np.linspace(a,b,(b-a)/Step)
    yn = InitVal
    for tn in tim:
        fn = F(tn, yn[:,-1])
        fn2 = F(tn+Step, yn[:,-1]+Step*fn)
        yn=np.append(yn, yn[:,-1]+ .5*Step*(fn+fn2),axis=1)
    return yn
