import numpy as np
from matplotlib import pyplot as plt

'''if __name__ == '__main__':



    i = int(input("number of data series"))
    fname = input("filename")
    for n in range(1,i+1):
        filename = f"{fname}{n}"
        file = open(f"{filename}.txt", "r")
        saveloc = "C:/Users/jwbar/Documents/Uni/6.sem/Bachelor/Figures/LTplots/"

        # reading the file
        data = file.read()

        d = data.replace('\n', ' ').replace('\t', ' ').split(" ")
        t = np.array(d[8::2], dtype = float)
        V = np.array(d[9::2], dtype = float)
        if len(t) < len(V):
            plen = len(t)
        else:
            plen = len(V)

        plt.figure(figsize=(4, 4), dpi=200)
        plt.plot(t, V, 'r')
        plt.xlabel('Time [s]')
        plt.ylabel(f'{d[1]}')
        plt.ylim([-13, 13])
        plt.xlim([0, t[len(t) - 1]])
        plt.grid(True)
        plt.subplots_adjust(left=0.15)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{saveloc}_{filename}_{d[4]}_time.svg")

        file.close()'''



if __name__ == '__main__':

    filename = input("Filename")

    file = open(f"LT/{filename}.txt", "r")
    fileSettings = open(f"LT/{filename}_settings.txt", "r")
    fileloc = "C:/Users/jwbar/Documents/Uni/6semBachelor/Bachelor/Python/BSc_Johan_Wilhelm_Bartling_s203824/scripts/"
    saveloc = "C:/Users/jwbar/Documents/Uni/6semBachelor/Bachelor/Figures/LTplots/"

    # reading the file
    data = file.read()
    sdata = fileSettings.read()
    d = data.replace('\n', ' ').replace('\t', ' ').replace(',', ' ').split(" ")
    sd = sdata.split("\n")
    dif = 0
    difd = 0
    if sdata[0] == '3':
        dif = 2
        difd = 1
    legend = []


    #t = np.array(d[3+dif::3+difd], dtype = float)*(float(sd[2].split(' ')[1]))
    t = np.array(d[3 + dif::3 + difd], dtype=float) * (float(sd[2].split(' ')[1]))
    S1 = np.array(d[4+dif::3+difd], dtype=float)
    legend.append(sd[4])
    S2 = 0
    if int(sdata[0]) >= 2:
        S2 = np.array(d[5+dif::3+difd], dtype=float)#15
        legend.append(sd[5])
    V3=0
    if int(sdata[0]) >= 3:
        V3 = np.array(d[6::4], dtype=float)
        legend.append(sd[6])


    if len(t) < len(S1):
        plen = len(t)
    else:
        plen = len(S1)

    plt.figure(figsize=(4, 3), dpi=200)
    if int(sdata[0]) >= 2:
        plt.plot(t, S2, 'r')
    else:
        plt.plot(t, S1, 'r')
    if int(sdata[0]) >= 2:
        plt.plot(t, S1, 'g')
    if int(sdata[0]) >= 3:
        plt.plot(t, V3, 'm')
    plt.xlabel(sd[2].split(' ')[0].replace('.',' '))
    plt.ylabel(sd[3].split(' ')[0].replace('.',' '))
    plt.title("LTSpice Simulation")
    plt.xlim(list(map(float,sd[1].split(' ')[0].split(','))))
    plt.ylim(list(map(float,sd[1].split(' ')[1].split(','))))
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(legend, loc= 'upper right')
    plt.subplots_adjust(left=0.17)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{saveloc}_{filename}_time.svg")

    if sdata[0] != '1' and S2 is not int:
        plt.figure(figsize=(4, 4), dpi=200)
        plt.plot(S1, S2, 'r')
        plt.xlabel(sd[9])
        plt.ylabel(sd[10])
        plt.title("LTSpice Simulation")
        if S1.min()< -12 or S1.max()>12:
            plt.xlim([-15,15])
        else:
            plt.xlim([S1.min()+(S1.min()/3), S1.max()+(S1.max()/3)])

        if S2.min()< -12 or S2.max()>12:
            plt.ylim([-15,15])
        else:
            plt.ylim([S2.min()+(S2.min()/3), S2.max()+(S2.max()/3)])
        plt.grid(visible=True, which='major', axis='both')
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{saveloc}_{filename}_phase.svg")

    file.close()