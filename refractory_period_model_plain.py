#!/usr/bin/env python
# coding: utf-8

# In[26]:


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.integrate import odeint


# In[27]:


# h < 0
# excitatory: J > 0
# inhibitory" J < 0

def vectorfield(states, t, params):
    """
    Defines the differential equations for the system (eqn 2.5) 

    Arguments:
        states :  vector of the state variables:
                  states = [pi_Q, pi_A]
        t :  time
        params :  vector of the parameters:
                  params = [h, J, p_RQ, p_AR]
    """
    pi_Q, pi_A = states
    h, J, p_RQ, p_AR = params
    
    p_QA = math.exp(h+(J*pi_A))/(1 + math.exp(h+(J*pi_A)))
    
    # Create f = (pi_Q', pi_A'):
    f = [(1 - pi_A - pi_Q)*(p_RQ) - (pi_Q * p_QA),
        (pi_Q*p_QA) - (pi_A*p_AR)]
    return f


# In[34]:


# parameter values
p_AR = 0.8
p_RQ = 0.01
h = -4
J = 300

# ODE solver parameters
# step size must = 1 to transform to discrete DE 
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 1000.0
numpoints = 1000

# create the time samples for the output of the ODE solver
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]


initial_pi_Q = np.arange(0,1.1,0.1)
initial_pi_A = initial_pi_Q[::-1]


# initial conditions
pi_A = 0.025
pi_Q = 1 - pi_A

# pack up parameters and initial conditions
params = [h, J, p_RQ, p_AR]
states_int = [pi_Q, pi_A]

# Call the ODE solver.
wsol = odeint(vectorfield, states_int, t, args=(params,),
              atol=abserr, rtol=relerr)

wsol_trans = wsol.T

plt.figure()
plt.plot(t,wsol_trans[0],label='pi_Q')
plt.plot(t,wsol_trans[1],label='pi_A')
plt.xlim(200,400)
#plt.ylim(0,0.2)
plt.title('pi_Q(t=0) = ' + str(pi_Q) + '     pi_A(t=0) = ' + str(pi_A)          + '\nh = ' + str(h) + '     J = ' + str(J))
plt.xlabel('t')
plt.legend()
plt.show()


# In[ ]:




