'''
CS 425: Brain Inspired Computing ~ hw 1
Professor Michmizos
Spring 2019
Anirudh Tunoori netid: at813

modelsim.py is a python script that simulates the firing of
three different neuron models, under varying parameters.
Interpreting these models, in the form of the outputted plots
can help distinguish the key features and aspects of these models.
'''
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint as oint
plt.style.use('fivethirtyeight')
#%matplotlib inline

#This function takes in the relevant parameters and simulates the Leaky Integrate-and-Fire model
#The output is a plot that displays the firing behavior of the membrane potential of the neuron.
def lif(Cm,Rm,Vt):
    print("lif")
    Vr = -70 #The resting potential of the membrane (rest voltage): -70 mv
    # Set up the time axis. Everything is modeled with respect to the time.
    T = 100 #msecs
    dt = .1 #delta t of .1 msecs
    delta_t = 0
    time = np.arange(0,T+dt, dt) # This array contains the x-coordinate (time) of each plotted point
    Vm = np.zeros(len(time)) # This array contains the y-coordinate (membrane potential) of each plotted point
    Vm[0] = Vr #Initialize the membrane potential at time 0 to the resting potential
    tau_m = Rm*Cm
    t_ref= 4 #The refractory period after an action potential
    spike = 1 #The peak spike of the action potential should reach around 30 mv

    #Set up the input current/stimulus I (Amperes) in the form of an array
    I = np.zeros(len(time))
    for i in range(0, len(time)):
        if i < 250:
            I[i] = 0
        elif i < 500:
            I[i] = 0.45 * 0.97 * (4 / 3) * math.pi  # I chose a constantly increasing stimulus
        elif i < 750:
            I[i] = 0.85 * 0.97 * (4 / 3) * math.pi
        else:
            I[i] = 0.97 * (4/3) * math.pi
        Vm[i] = Vr

    # Obtain the membrane potential over this time frame:
    for x,t in enumerate(time):
        if t > 0 and t > delta_t:
            Vm[x] = Vm[x-1] + (1/(tau_m*dt))*(Rm*I[x] - Vm[x-1]) #Integrate
            if Vm[x] >= Vt: #Spike!
                Vm[x] += spike
                delta_t = t + t_ref

    #Plot the behavior in accordance with this model
    plt.figure(figsize=(7,5))
    l1 = plt.plot(time,Vm,'b-', linewidth = .5,label="Membrane Potential (mV)")
    l2 = plt.plot(time,I,'g-',label="Input Current (A)")
    plt.title("Leaky Integrate-and-Fire model")
    plt.ylabel("Membrane Potential (mV), Input Current (A)")
    plt.xlabel("Time (msec)")
    plt.legend(loc="upper right")
    plt.ylim([-80,50])
    plt.show(block=False)

    #We can also plot the firing rate as a function of the input current
    Fr = np.zeros(len(time))
    for t2 in range(0, len(time)):
        if I[t2] < Vt/Rm or I[t2] == 0:
            Fr[t2] = 0
        else:
            Fr[t2] = (-10)/(t_ref - tau_m*math.log10(1 - Vt/(I[t2]*Rm)))
    plt.figure(figsize=(7, 5))
    l3 = plt.plot(time, Fr, 'm-', label="Firing Rate (.01 sec^-1)")
    l2 = plt.plot(time, I, 'g-', label="Input Current (A)")
    plt.title("Leaky Integrate-and-Fire model: Firing Rate")
    plt.ylabel("Firing Rate (.01 sec^-1), Input Current (A)")
    plt.xlabel("Time (msec)")
    plt.legend(loc="upper right")
    plt.show(block=False)

    return

#This function takes in the relevant parameters and simulates the Izhikevich model.
#The output is a plot that displays the firing behavior of the membrane potential of the neuron.
def izhikevich(Input, a, b, c, d, spike):
    print("izhikevich")
    Vr = -70  # The resting potential of the membrane (rest voltage): -70 mv
    # Set up the time axis. Everything is modeled with respect to the time.
    T = 100  # msecs
    dt = .1  # delta t of .1 msecs
    time = np.arange(0, T + dt, dt)  # This array contains the x-coordinate (time) of each plotted point
    Vm = np.zeros(len(time))  # This array contains the y-coordinate (membrane potential) of each plotted point
    Vm[0] = Vr  # Initialize the membrane potential at time 0 to the resting potential
    Um = np.zeros(len(time))  # This array represents the membrane recovery
    Um[0] = -15  # Initialize the recovery history

    I = np.zeros(len(time)) # Set up the Input Current (Ampere) for the course of this time frame
    for i in range(0, len(time)):
        if i < 250:
            I[i] = 0
        elif i < 500:
            I[i] = 0.45 * Input  # I chose a constantly increasing stimulus
        elif i < 750:
            I[i] = 0.75 * Input
        else:
            I[i] = Input

    # Obtain the membrane potential over this time frame:
    for t in range(1, len(time)):
        if Vm[t-1] < spike:
            #Set up the system of Linear Differential Equations:
            dv = (0.04*Vm[t-1] + 5)*(Vm[t-1]) - Um[t-1] + 140
            Vm[t] = Vm[t-1] + I[t-1]*dt + dv*dt
            du = a*(b*Vm[t-1] - Um[t-1])
            Um[t] = Um[t-1] + du*dt
        else:
            Vm[t-1] = spike #Set to the spike
            Vm[t] = c   #Return to the resting membrane potential
            Um[t] = Um[t-1] + d #Reset the membrane recovery

    #Plot the simulation
    plt.figure(figsize=(7, 5))
    l1 = plt.plot(time, Vm, 'b-', linewidth=.5, label="Membrane Potential (mV)")
    l2 = plt.plot(time, I, 'g-', label="Input Current (A)")
    plt.title("Izhikevich model")
    plt.ylabel("Membrane Potential (mV), Input Current (A)")
    plt.xlabel("Time (msec)")
    plt.legend(loc="upper right")
    plt.ylim([-80, 50])
    plt.show(block=False)

    return


#This function takes in the relevant parameters and simulates the Hodgkin-Huxley model.
#The output is a plot that displays the firing behavior of the membrane potential of the neuron.
def hodgkin_huxley(Input, Cm, gNa, gK, gL, VNa, VK, VL):
    print("hodgkin_huxley")
    Vr = -70  # The resting potential of the membrane (rest voltage): -70 mv
    # Set up the time axis. Everything is modeled with respect to the time.
    T = 100  # msecs
    dt = .1  # delta t of .1 msecs
    time = np.arange(0, T + dt, dt)  # This array contains the x-coordinate (time) of each plotted point

    #Source for obtaining the alpha, beta functions for potassium channel activation,
    # sodium channel activation, and sodium channel inactivation:
    # https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
    m_a = lambda V: 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)) #sodium channel activation
    n_a = lambda V: 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)) #potassium  channel activation
    h_a = lambda V: 0.07 * np.exp(-(V + 65.0) / 20.0) #sodium channel inactivation
    m_b = lambda V: 4.0 * np.exp(-(V + 65.0) / 18.0)
    n_b = lambda V: 0.125 * np.exp(-(V + 65) / 80.0)
    h_b = lambda V: 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    #Using the primary equation for this model, we can come up with equations for the current given the membranes
    #voltage potential
    INa = lambda V, m, h: gNa * m ** 3 * h * (V - VNa)
    IK = lambda V, n: gK * n ** 4 * (V - VK)
    IL = lambda V: gL * (V - VL)


    current = np.zeros(len(time))  # Set up the Input Current (Ampere) for the course of this time frame
    for i in range(0, len(time)):
        if i < 50:
            current[i] = 0
        elif i < 300:
            current[i] = i * Input * .05
        elif i < 750:
            current[i] = .75 * Input
        else:
            current[i] = Input


    def I(t):
        if t < 50:
            return 0
        elif t < 300:
            return t * 14 *.05
        elif t < 750:
            return .75 * 14
        else:
            return 14
        return

    # Define The Differential Equation for the HH model
    def DV(independents, t):
        V, m, h, n = independents
        dV = (I(t) - INa(V,m,h) - IK(V,n) - IL(V)) / Cm
        dm = m_a(V) * (1.0 - m) - m_b(V) * m
        dh = h_a(V) * (1.0 - h) - h_b(V) * h
        dn = n_a(V) * (1.0 - n) - n_b(V) * n
        return dV, dm, dh, dn

    x = oint(DV,[-70, 0.05, 0.6, 0.30], time) #Solve and obtain V
    V = x[:,0]


    # Plot the simulation
    plt.figure(figsize=(7, 5))
    l1 = plt.plot(time, V, 'b-', linewidth=.5, label="Membrane Potential (mV)")
    l2 = plt.plot(time, current, 'g-', label="Input Current (A)")
    plt.title("Hodgkin-Huxley model")
    plt.ylabel("Membrane Potential (mV), Input Current (A)")
    plt.xlabel("Time (msec)")
    plt.legend(loc="upper right")
    plt.ylim([-80, 120])
    plt.show(block=False)

    return

if __name__ == "__main__":

    # For each of these models, we can modify and vary the default parameters
    lif(Cm = 10,Rm = 0.7142857142,Vt = -65)
    izhikevich(Input = 10,a = 0.02,b = 0.25,c = -65,d = 6, spike = 30)
    hodgkin_huxley(Input = 12, Cm = 10, gNa = 120.0, gK = 39.0, gL = 0.42, VNa = 80.1, VK = -60.09, VL = -34.14)
    plt.show()
