import numpy as np
import matplotlib.pyplot as plt

x0=0
h = 0.1
t = np.arange(0, 1 + h, h)
x=[]

def force(x,t):
    k = 1
    force = -k * x[t]
    return force

s = np.zeros(len(t))




for i in range(0, len(t) - 1):


    force = force(x,t)
    x[i+1] = s[i]+h*force[x[i], t[i]]
    x.append(x[i+1])

    

def show_plots():
    plt.plot(x,t)
    plt.show

show_plots()