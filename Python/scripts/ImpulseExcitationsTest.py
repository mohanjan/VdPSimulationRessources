# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
#----------------------------------------
#Imports from Marcus scripts
import non_lin_tools as nlt
from scipy.integrate import odeint
import torch as pt
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq
import time
import os
#----------------------------------------
if __name__ == '__main__':
    R1 = 4700
    C1 = 10**-7
    C2 = 10**-9
    R5 = 10000
    R6 = 4700
    R7 = 1000

    # Variable Component values
    R3 = 220
    R4 = 10000
    R2 = 1000

    sysnum = 4
    system = [None]*sysnum

    saveloc = "C:/Users/jwbar/Documents/UNI/6semBachelor/Bachelor/Figures/"

    # constraints:
    #W0 = math.sqrt(R6/(R2 * R5 * R7 * C1 * C2))
    model = 'VDP'
    N = 1
    #W0 = math.sqrt(R6 / (R2 * R5 * R7 * C1 * C2))
    #f_min = W0/(np.pi*2.0)
    f_min = 1000
    f_max = f_min
    d = 0.0
    k = 0.0
    #mu = R6/(R3*100*W0*C1*R5)
    mu = [0.01, 0.5, 1, 5]
    v = 0.0
    P = (d, k, mu, v)  # input vector
    tSim = 10.0  # How long to run simulation
    sFreq = 88000.0  # sampling frequency
    drvP = [None]*sysnum
    E0 = [None]*sysnum
    driver = 'impulse'

    for i in range(sysnum):
        #E0[i] = R6 / (R4 * R5 * R7 * C1 * C2 * (W0 ** 2))
        #drvP[i] = [['impulse', -1*E0[i], 0.0, tSim]]
        drvP[i] = [['impulse', 1, 0.0, tSim]]
        system[i] = nlt.NonLinMdl(mdlName=model,
                           N=N, fMin=f_min,
                           fMax=f_max,
                           P= (0,0,mu[i],0),
                           drvP=drvP[i],
                           tSim=tSim,
                           sFreq=sFreq)
    #for i in range(sysnum):
    #    system[i].solveSystem()
    #    system[i].saveSol()



    sysname = [None]*sysnum
    # series 3
    sysname[0] = "VDP_N=1_f=[2326.26_2326.26]_P=[0_0_0.0146163_0_]_Tsim=10_Fs=44000_Drv=[impulse_-0.022_0.0_10.0_]"

    #series 4
    #44kS
    sysname[1] = "VDP_N=1_f=[10911.1_10911.1]_P=[0_0_0.00311621_0_]_Tsim=10_Fs=44000_Drv=[impulse_-0.10000000000000002_0.0_10.0_]"

    #mu values 0.01 -> 0.5 -> 1->5
    #sysname[0] = "VDP_N=1_f=[1000_1000]_P=[0_0_0.01_0_]_Tsim=10_Fs=88000_Drv=[impulse_1_0.0_10.0_]"
    #sysname[1] = "VDP_N=1_f=[1000_1000]_P=[0_0_0.5_0_]_Tsim=10_Fs=88000_Drv=[impulse_1_0.0_10.0_]"
    #sysname[2] = "VDP_N=1_f=[1000_1000]_P=[0_0_1_0_]_Tsim=10_Fs=88000_Drv=[impulse_1_0.0_10.0_]"
    #sysname[3] = "VDP_N=1_f=[1000_1000]_P=[0_0_5_0_]_Tsim=10_Fs=88000_Drv=[impulse_1_0.0_10.0_]"

    slocA = f"{driver}/"
    try:
        os.mkdir(f"{saveloc}{slocA}")
    except FileExistsError:
        print("Directory already exists")
    series = 4
    #plt.figure(figsize=(5,5))
    #plt.title(f"Simulation of different \u03BC values")
    for i in range(sysnum):
        if sysname[i] is not None:
            system[i].loadSol(fName=sysname[i])
    #    plt.subplot(2,2,i+1)
    #    plt.plot(system[i].sol_y.T[60000:64000, 0], system[i].sol_y.T[60000:64000, 1], 'r')
    #    plt.title(f"{i+1}")

    #plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.3)
    #plt.savefig(f"{saveloc}{slocA}_muplot.svg")

    fnameA = f"Series{series}{i}"
    lim = 80000
    limd = 100
    plt.figure(figsize=(4, 4), dpi=200)
    plt.plot(system[i].sol_t[lim:lim + limd], system[i].sol_y.T[lim:lim + limd, 0], 'r')
    plt.xlim([system[i].sol_t[lim], system[i].sol_t[lim + limd]])
    plt.xlabel('Sample time [s]')
    plt.ylabel('Amplitude p.u.')
    plt.grid(True)
    plt.title("Numerical Model")
    plt.subplots_adjust(left=0.17)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{saveloc}{slocA}{fnameA}_W0_{int(W0)}_mu_{int(mu)}_time.svg")

    plt.figure(figsize=(4, 4), dpi=200)
    plt.plot(system[i].sol_y.T[10000:20000,0], system[i].sol_y.T[10000:20000, 1], 'r')
    plt.xlim([system[i].sol_y[1].min(), system[i].sol_y[1].max()])
    plt.xlabel('Amplitude p.u.')
    plt.ylabel('Amplitude p.u. derivative')
    plt.grid(True)
    plt.title("Numerical Model")
    plt.subplots_adjust(left=0.17)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{saveloc}{slocA}{fnameA}_W0_{int(W0)}_mu_{int(mu)}_phase.svg")

#system.plotTempAmplitudes(vmin=-12, vmax=12)