import non_lin_tools as nlt
from scipy.integrate import odeint
import torch as pt
import numpy as np
from matplotlib import pyplot as plt
import time
import os



def plottingtimeseries(data, sdata):
    d = data.replace('\n', ' ').replace('\t', ' ').replace(',', ' ').split(" ")
    sd = sdata.split("\n")
    dif = 0
    difd = 0
    if sdata[0] == '2':
        dif = 0
        difd = 0
    elif sdata[0] == '3':
        dif = 2
        difd = 1
    legend = []

    inc = float(d[10 + dif])  # 10/12
    X = np.array(d[11 + dif::4 + difd], dtype=float)  # 11::4/13::5
    t = X * inc * (float(sd[2].split(' ')[1]))
    V1 = np.array(d[12 + dif::4 + difd], dtype=float)  # 14
    legend.append(sd[4])
    V2 = 0
    if int(sdata[0]) >= 2:
        V2 = np.array(d[13 + dif::4 + difd], dtype=float)  # 15
        legend.append(sd[5])
    V3 = 0
    if int(sdata[0]) >= 3:
        V3 = np.array(d[16::5], dtype=float)
        legend.append(sd[6])

    plt.plot(t, V1, 'r')
    if int(sdata[0]) >= 2:
        plt.plot(t, V2, 'g')
    if int(sdata[0]) >= 3:
        plt.plot(t, V3, 'm')
    plt.xlabel(sd[2].split(' ')[0].replace('.', ' '))
    plt.ylabel(sd[3])
    plt.xlim([t[0], t.max()])
    plt.grid(visible=True, which='major', axis='both')

    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(legend, loc='upper right')

def plottingphasespace(data, sdata):
    d = data.replace('\n', ' ').replace('\t', ' ').replace(',', ' ').split(" ")
    sd = sdata.split("\n")
    dif = 0
    difd = 0
    if sdata[0] == '2':
        dif = 0
        difd = 0
    elif sdata[0] == '3':
        dif = 2
        difd = 1
    legend = []

    inc = float(d[10 + dif])  # 10/12
    X = np.array(d[11 + dif::4 + difd], dtype=float)  # 11::4/13::5
    t = X * inc * (float(sd[2].split(' ')[1]))
    V1 = np.array(d[12 + dif::4 + difd], dtype=float)
    # 14
    legend.append(sd[4])
    V2 = 0
    if int(sdata[0]) >= 2:
        V2 = np.array(d[13 + dif::4 + difd], dtype=float)
    # 15
    legend.append(sd[5])
    V3 = 0
    if int(sdata[0]) >= 3:
        V3 = np.array(d[16::5], dtype=float)

    legend.append(sd[6])
    plt.plot(V1, V2, 'r')
    plt.xlabel(sd[9])
    plt.ylabel("Voltage V1[V]")
    plt.grid(visible=True, which='major', axis='both')
    plt.title("Oscilloscope Measurements")
    file.close()
    fileSettings.close()

'''
if __name__ == '__main__':

    i = input("number of data series")
    filename = f"NF{i}"
    file = open(f"Traces/{filename}.csv", "r")
    fileSettings = open(f"Traces/{filename}_settings.txt", "r")
    fileloc = "C:/Users/jwbar/Documents/Uni/6semBachelor/Bachelor/Python/BSc_Johan_Wilhelm_Bartling_s203824/scripts/Traces/"
    saveloc = "C:/Users/jwbar/Documents/Uni/6semBachelor/Bachelor/Figures/OscPlot/"
    data = file.read()
    sdata = fileSettings.read()
    # plotting time series
    plt.figure(1)
    plt.title("Oscilloscope Measurements")
    plottingtimeseries(data, sdata)
    plt.subplots_adjust(top=0.9)
    plt.subplots_adjust(left=0.1)
    plt.savefig(f"{saveloc}_{filename}_time.svg")
    plt.close()

    # plotting phase space
    plt.figure(2)
    plt.title("Oscilloscope Measurements")
    plottingphasespace(data, sdata)
    plt.subplots_adjust(top=0.9)
    plt.subplots_adjust(left=0.1)
    plt.savefig(f"{saveloc}_{filename}_phase.svg")

    file.close()

'''

if __name__ == '__main__':
# pairs 6,3; 5,4
    i = input("number of data series")
    j = input("number of data series")
    filename1 = f"NF{i}"
    filename2 = f"NF{j}"
    file1 = open(f"Traces/{filename1}.csv", "r")
    file2 = open(f"Traces/{filename2}.csv", "r")
    fileSettings1 = open(f"Traces/{filename1}_settings.txt", "r")
    fileSettings2 = open(f"Traces/{filename2}_settings.txt", "r")
    fileloc = "C:/Users/jwbar/Documents/Uni/6semBachelor/Bachelor/Python/BSc_Johan_Wilhelm_Bartling_s203824/scripts/Traces/"
    saveloc = "C:/Users/jwbar/Documents/Uni/6semBachelor/Bachelor/Figures/OscPlot/"
    data1 = file1.read()
    sdata1 = fileSettings1.read()
    data2 = file2.read()
    sdata2 = fileSettings2.read()
    plt.figure(1)
    plt.subplot(211)
    plt.title("Oscilloscope Measurements")
    plottingtimeseries(data1, sdata1)
    plt.subplots_adjust(top=1-0.1)
    plt.subplots_adjust(left=0.1)
    # reading the file
    plt.subplot(212)
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(bottom=0.1)
    plottingtimeseries(data2, sdata2)
    plt.savefig(f"{saveloc}_{filename1}_{filename2}_stacked_time.svg")
    plt.close()
