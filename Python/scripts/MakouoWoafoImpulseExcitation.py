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
    R1 = 790
    C1 = 10**-7
    C2 = 10**-8
    R5 = 10**4
    R6 = 10**4
    R7 = 10**4

    # Variable Component values
    R3 = 220
    R4 = [5000, 10000, 20000]
    R2 = 10**4

    system = [None]*4

    saveloc = "C:/Users/jwbar/Documents/UNI/6semBachelor/Bachelor/Figures/"

    # constraints:
    W0 = math.sqrt(R6/(R2 * R5 * R7 * C1 * C2))
    model = 'VDP'
    N = 1
    W0 = math.sqrt(R6 / (R2 * R5 * R7 * C1 * C2))
    f_min = W0/(np.pi*2.0)
    f_max = f_min
    d = 0.0
    k = 0.0
    mu = R6/(R3*100*W0*C1*R5)  # non-linearity factor
    v = 0.0
    P = (d, k, mu, v)  # input vector
    tSim = 10.0  # How long to run simulation
    sFreq = 40000.0  # sampling frequency
    drvP = [None]*3
    E0 = [None]*3

    driver = 'impulse'
    #drvP[0] = [['impulse', 1, 0.0, tSim]]

    for i in range(3):
        E0[i] = R6 / (R4[i] * R5 * R7 * C1 * C2 * (W0 ** 2))
        drvP[i] = [['impulse', -1*E0[i], 0.0, tSim]]
        system[i] = nlt.NonLinMdl(mdlName=model,
                           N=N, fMin=f_min,
                           fMax=f_max,
                           P=P,
                           drvP=drvP[i],
                           tSim=tSim,
                           sFreq=sFreq)
    #for i in range(3):
    #    system[i].solveSystem()
    #    system[i].saveSol()



    sysname = [None]*3
    # series 1
    #sysname[0] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_4.0542_0_]_Tsim=10_Fs=40000_Drv=[impulse_-1.9999999999999996_0.0_10.0_]"
    #sysname[1] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_4.0542_0_]_Tsim=10_Fs=40000_Drv=[impulse_-0.9999999999999998_0.0_10.0_]"
    #sysname[2] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_4.0542_0_]_Tsim=10_Fs=40000_Drv=[impulse_-0.4999999999999999_0.0_10.0_]"

    #series 2
    sysname[0] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_0.14374_0_]_Tsim=10_Fs=40000_Drv=[impulse_-1.9999999999999996_0.0_10.0_]"
    sysname[1] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_0.14374_0_]_Tsim=10_Fs=40000_Drv=[impulse_-0.9999999999999998_0.0_10.0_]"
    sysname[2] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_0.14374_0_]_Tsim=10_Fs=40000_Drv=[impulse_-0.4999999999999999_0.0_10.0_]"



    slocA = f"{driver}/mu{round(mu, 1)}/"
    try:
        os.mkdir(f"{saveloc}{slocA}")
    except FileExistsError:
        print("Directory already exists")
    series = 2
    for i in range(3):
        if sysname[i] is not None:
            system[i].loadSol(fName=sysname[i])

        fnameA = f"Series{series}{i}"
        lim = 80000
        limd = 400
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

        plt.figure(figsize=(4, 3), dpi=200)
        plt.plot(system[i].sol_y.T[0:2000,0], system[i].sol_y.T[0:2000, 1], 'r')
        plt.xlim([system[i].sol_y[1].min(), system[i].sol_y[1].max()])
        plt.xlabel('Amplitude p.u.')
        plt.ylabel('Amplitude p.u. derivative')
        plt.grid(True)
        plt.title("Numerical Model")
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{saveloc}{slocA}{fnameA}_W0_{int(W0)}_mu_{int(mu)}_phase.svg")

        #system.plotTempAmplitudes(vmin=-12, vmax=12)'''