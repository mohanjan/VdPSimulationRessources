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

    C1 = 10**-7
    C2 = 10**-9
    R1 = 4700
    R2 = 1000
    R3 = 220
    R4 = 10000
    R5 = 10000
    R6 = 4700
    R7 = 1400

    saveloc = "C:/Users/jwbar/Documents/UNI/6semBachelor/Bachelor/Figures/"

    system = [None] * 3
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
    sFreq = 96000.0  # sampling frequency
    drvP = [None]*3
    E0 = R6 / (R4 * R5 * R7 * C1 * C2 * (W0 ** 2))

    driver = 'sin'
    sf = [8500,9000,9500]

    for i in range(3):
        drvP[i] = [['impulse', 1, 0.0, 1.0], [driver, 10*E0, sf[i], 1.0, tSim]]
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
    #series 3

    #sysname[0] = "VDP_N=1_f=[3450.4_3450.4]_P=[0_0_0.00985431_0_]_Tsim=10_Fs=44000_Drv=[impulse_1_0.0_1.0_sin_0.5000000000000001_8500_1.0_10.0_]"
    #sysname[1] = "VDP_N=1_f=[3450.4_3450.4]_P=[0_0_0.00985431_0_]_Tsim=10_Fs=44000_Drv=[impulse_1_0.0_1.0_sin_0.5000000000000001_9000_1.0_10.0_]"
    #sysname[2] = "VDP_N=1_f=[3450.4_3450.4]_P=[0_0_0.00985431_0_]_Tsim=10_Fs=44000_Drv=[impulse_1_0.0_1.0_sin_0.5000000000000001_9500_1.0_10.0_]"

    #96kS
    #sysname[0] = "VDP_N=1_f=[2326.26_2326.26]_P=[0_0_0.0146163_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_0.022_2000_1.0_10.0_]"
    #sysname[1] = "VDP_N=1_f=[2326.26_2326.26]_P=[0_0_0.0146163_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_0.022_2300_1.0_10.0_]"
    #sysname[2] = "VDP_N=1_f=[2326.26_2326.26]_P=[0_0_0.0146163_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_0.022_2500_1.0_10.0_]"

    #Series 4
    #sysname[0] = "VDP_N=1_f=[9221.57_9221.57]_P=[0_0_0.00368715_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_0.10000000000000002_8500_1.0_10.0_]"
    #sysname[1] = "VDP_N=1_f=[9221.57_9221.57]_P=[0_0_0.00368715_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_0.10000000000000002_9000_1.0_10.0_]"
    #sysname[2] = "VDP_N=1_f=[9221.57_9221.57]_P=[0_0_0.00368715_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_0.10000000000000002_9500_1.0_10.0_]"

    sysname[0] = "VDP_N=1_f=[9221.57_9221.57]_P=[0_0_0.00368715_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_1.0000000000000002_8500_1.0_10.0_]"
    sysname[1] = "VDP_N=1_f=[9221.57_9221.57]_P=[0_0_0.00368715_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_1.0000000000000002_9000_1.0_10.0_]"
    sysname[2] = "VDP_N=1_f=[9221.57_9221.57]_P=[0_0_0.00368715_0_]_Tsim=10_Fs=96000_Drv=[impulse_1_0.0_1.0_sin_1.0000000000000002_9500_1.0_10.0_]"


    slocA = f"{driver}/"

    try:
        os.mkdir(os.path.join(saveloc, slocA))
    except FileExistsError:
        None

    for i in range(3):
        system[i].loadSol(fName=sysname[i])
        fnameAA = ""
        fnameA = f"{fnameAA}_{i}_sin"
        lim = int(sFreq)
        plt.figure(figsize=(4, 4), dpi=200)
        plt.plot(system[i].sol_t[4*lim:4*lim+2000], system[i].sol_y.T[4*lim:4*lim+2000, 0], 'r')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude p.u.')
        plt.title("Python Simulation")
        plt.grid(True)
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{saveloc}{slocA}{fnameA}_W0_{int(W0)}_mu_{int(mu)}_time.svg")

        plt.figure(figsize=(4, 4), dpi=200)
        plt.plot(system[i].sol_y.T[lim:lim+2000,0], system[i].sol_y.T[lim:lim+2000, 1], 'r')
        plt.xlabel('Amplitude p.u.')
        plt.ylabel('diff Amplitude p.u.')
        plt.grid(True)
        plt.title("Python Simulation")
        plt.subplots_adjust(left=0.15)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{saveloc}{slocA}{fnameA}_W0_{int(W0)}_mu_{int(mu)}_phase.svg")

        #system.plotTempAmplitudes(vmin=-12, vmax=12)'''