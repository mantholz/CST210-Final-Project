"""
Created on Sat Mar 22

@author: mantholz
"""
import scipy.integrate as si
import numpy as np
import matplotlib.pylab as plt
# Create a function that defines the thrust
def Thrust(t):    
#   F1 = Thrust used at intial take off and all the way till max thrust is reached
#   F2 = Thrust that starts at max thrust and decreases to the constant thrust
#   F3 = Thrust that is constant
    Fm = 3.29 # Max force
    Fs = 2.63 # Constant force
    Tm = .54 # Time max force was reached
    Ts = .78 # Time constant force was reached
    Te = 1.4
    F1 = (Fm/Tm)*t
    s = (Fs-Fm)/(Ts-Tm)
    F2 = s*(t-Tm)+Fm
    F3 = Fs
    if t < Tm:
        th = F1
    elif t < Ts:
        th = F2
    elif t < Te:
        th = F3
    else:
        th = 0
    return th
# Create a function that defines the rhs of the differential equation system
def calc_RHS(y,t,p):
#   y = a list that contains the system state
#   t = the time for which the right-hand-side of the system equations
#       is to be calculated.
#   p = a tuple that contains any parameters needed for the model
#   Unpack the state of the system
    y0 = y[0]
    y1 = y[1]

#   Unpack the parameter list
    g = 9.81
    th = Thrust(t)
    rho = 1.225 # Desity of air at stp
    Cd = .75 # Drag coefficient
    r = .0174625 # Radius of the cylinder
    A = np.pi*r**2#Area of the model rocket
    d = (.5*(rho*Cd*A*(y1**2)))*np.sign(y1)
    m,mr,me = p
    #   Calculate the rates of change (the derivatives)
    dy0dt = y1
    dy1dt = (1/m)*(th-d-(m*g))

    return [dy0dt,dy1dt]

# Define the initial conditions
y_0 = [0,0]

# Define the time grid
tg = np.linspace(0,6.5,100)

# Define the model parameters
mr = .0663
me = .0193
m = mr + me
p = m,mr,me

# Solve the DE
sol = si.odeint(calc_RHS,y_0,tg,args=(p,))
y0 = sol[:,0]
y1 = sol[:,1]

# Plot the solution
plt.plot(tg,y0,color='b')
plt.xlabel('t[s]', fontsize=14)
plt.ylabel('y[m]', fontsize=14)
plt.title('Model Rocket Trajectory With B6-2 Engine',fontsize=20,
          color='blue')
plt.grid(True)
plt.savefig('ModelRocketTrajectory(B6-2).png', dpi=300)
plt.show()

