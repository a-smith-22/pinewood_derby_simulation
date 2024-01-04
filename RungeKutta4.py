'''
==========================================================================================
TITLE: Fourth Order Runge-Kutta Method for Vehicle Runtime
AUTHOR: Andrew Smith
DATE: May 7th, 2023 (last updated)
==========================================================================================

'''

# 0 INPUT VARIABLES
# Car Properties
m = 142          # car mass [g]
A = 18004.40024  # surface area [mm^2]
d = 173.444262    # center of mass location [mm]

# Simulation results
Px = -9.216926e-6 # pressure force (N), form drag
Py = 1.331187e-5  # pressure force (N), lift
Vx = -1.360404e-4 # viscous force (N), friction drag
Vy = 1.573853e-6  # viscous force (N), vertical drag

# Simulation Input
h = 1e-6 # step size [s], minimize for accuracy

'''
==========================================================================================
'''

# 1 PYTHON IMPORT
import numpy as np
import pandas as pd 


# 2 INTIAL CALCULATIONS
# Constant Values
L1 = 2.13   # length of hill [m]
L2 = 6.40   # length of flat [m]
theta = 26  # angle of hill [deg]
g = 9.807   # gravity [m/s^2]
r = 0.0011  # car axial radius [m]
mu = 0.1    # friction coefficient for graphite-steel
rho = 1.205 # density of air at STP [kg/m^3]

# Convert inputs to SI units
m = m / 1000 # mass [kg]
A = A / (1000**2) # surface area [m^2]
d = d / 1000 # center of mass location [m]
 
# Compute aerodynamic coefficients
u_sim = 1.5 # airspeed used in simulation [m/s]
FD = Px+Vx # total drag force
FL = Py+Vy # total lift force
Cd = abs( 2*FD / (rho * u_sim**2 * A) ) # drag coefficient
CL = abs( 2*FL / (rho * u_sim**2 * A) )  # lift coefficient

print("AERODYNAMIC COEFFICIENTS")
print("Cd (drag coefficient): " + f"{Cd:0.9f}")
print("CL (lift coefficient): " + f"{CL:0.9f}")

# Simulation variables
H = (L1 + d)*np.sin(theta) # maximum initial starting height of car on top of the hill
th = np.sqrt(2*H/g) # compute the hill run time
u0 = np.sqrt(2*g*H) # speed of the car at the bottom of the hill


# 3 FOURTH ORDER RUNGE-KUTTA METHOD
# solve for u as function of time
def f(v):
    # differential equation to solve for, du/dt = f(u)
    # v = speed (u)
    c1 = (2/m)*r*mu*rho*A*CL # first term
    c2 = (1/(2*m))*rho*A*Cd  # second term
    c3 = 4*r*mu*g            # third term
    return ( (v**2)*(c1-c2)-c3 )

t_min = L2 / u0              # time assuming no drag
n_steps = int(2*t_min / h)   # maximum number of computation steps, double as a safety net
print("")
print("EXECUTION TIME (s): " + f"{(n_steps * 1.014994e-5):0.3f}" )
print("computing...")

u = u0  #define inital condition
D = 0   # distance on flat track
tf = 0  # time travelled on flat part

for i in range(n_steps):
    # Find current position and time
    D += u*h # add to current distance
    tf += h  # add to current time
    
    if D > L2: # once finish line has been reached
        break # stop computation

    # Find new speed
    k1 = h*f(u)
    k2 = h*f(u + k1/2)
    k3 = h*f(u + k2/2)
    k4 = h*f(u + k3)
    u = u + k1/6 + k2/3 + k3/3 + k4/6 # find new speed

print("")
print("SYSTEM OUTPUT:")
print("Hill time (s): " + f"{(th):0.6f}")
print("Flat time (s): " + f"{(tf):0.6f}")
print("Minimum  time (s): " + f"{(tf):0.6f}")
print("Efficiency of flat: " + f"{(100*t_min/tf):0.4f}" + "%")
print("")
print("TOTAL TIME (s): " + f"{(th+tf):0.6f}")