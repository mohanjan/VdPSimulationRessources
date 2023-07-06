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
    R3 = 7.8
    R4 = 10000
    R2 = 10**4

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
    sFreq = 40000.0  # sampling frequency
    drvP = [None]*3
    E0 = R6 / (R4 * R5 * R7 * C1 * C2 * (W0 ** 2))

    driver = 'sin'
    sf = [1.02, 1.5, 3.0]

    for i in range(3):
        drvP[i] = [['impulse', 1, 0.0, 0.03], [driver, 1.1*E0, 0.56, 0.5, tSim], [driver, 0.4*E0, sf[i], 0.5, tSim]]
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
    #series 1
    sysname[0] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_4.0542_0_]_Tsim=10_Fs=40000_Drv=[impulse_1_0.0_0.03_sin_1.0999999999999999_0.56_0.5_10.0_sin_0.3999999999999999_1.02_0.5_10.0_]"
    sysname[1] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_4.0542_0_]_Tsim=10_Fs=40000_Drv=[impulse_1_0.0_0.03_sin_1.0999999999999999_0.56_0.5_10.0_sin_0.3999999999999999_1.5_0.5_10.0_]"
    sysname[2] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_4.0542_0_]_Tsim=10_Fs=40000_Drv=[impulse_1_0.0_0.03_sin_1.0999999999999999_0.56_0.5_10.0_sin_0.3999999999999999_3.0_0.5_10.0_]"

    #series 2
    #sysname[0] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_0.14374_0_]_Tsim=10_Fs=40000_Drv=[impulse_1_0.0_0.03_sin_1.0999999999999999_0.56_0.5_10.0_sin_0.3999999999999999_1.02_0.5_10.0_]"
    #sysname[1] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_0.14374_0_]_Tsim=10_Fs=40000_Drv=[impulse_1_0.0_0.03_sin_1.0999999999999999_0.56_0.5_10.0_sin_0.3999999999999999_1.5_0.5_10.0_]"
    #sysname[2] = "VDP_N=1_f=[503.292_503.292]_P=[0_0_0.14374_0_]_Tsim=10_Fs=40000_Drv=[impulse_1_0.0_0.03_sin_1.0999999999999999_0.56_0.5_10.0_sin_0.3999999999999999_3.0_0.5_10.0_]"


    slocA = f"{driver}/"

    try:
        os.mkdir(os.path.join(saveloc, slocA))
    except FileExistsError:
        None
    series = 1
    for i in range(3):
        if sysname[i] is not None:
            system[i].loadSol(fName=sysname[i])

        fnameA = f"Series{series}_{i}_sin"
        lim = 160000
        limd = 1000
        plt.figure(figsize=(4, 4), dpi=200)
        plt.plot(system[i].sol_t[lim:lim+limd], system[i].sol_y.T[lim:lim+limd, 0], 'r')
        plt.xlim([system[i].sol_t[lim], system[i].sol_t[lim+limd]])
        plt.xlabel('Sample time [s]')
        plt.ylabel('Amplitude p.u.')
        plt.title("Numerical model")
        plt.grid(True)
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{saveloc}{slocA}{fnameA}_W0_{int(W0)}_mu_{int(mu)}_time.svg")

        plt.figure(figsize=(4, 3), dpi=200)
        plt.plot(system[i].sol_y.T[0:10000,0], system[i].sol_y.T[0:10000, 1], 'r')
        plt.xlim([system[i].sol_y[1].min(), system[i].sol_y[1].max()])
        plt.xlabel('Amplitude p.u.')
        plt.ylabel('Amplitude p.u. derivative')
        plt.title("Numerical model")
        plt.grid(True)
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{saveloc}{slocA}{fnameA}_W0_{int(W0)}_mu_{int(mu)}_phase.svg")

        #system.plotTempAmplitudes(vmin=-12, vmax=12)'''