import math
import numpy

if __name__ == '__main__':

    C1 = 10**-7
    C2 = 10**-9
    R1 = 4700
    R2 = 1000
    R3 = R1/100
    R3 = 220
    R4 = 10000
    R5 = 10000
    R6 = 4700
    R7 = 1000

    # Variable Component values


    W0 = math.sqrt(R6 / (R2 * R5 * R7 * C1 * C2))
    f = W0 / (2 * numpy.pi)
    mu = R6 / (R3 * 100 * W0 * C1 * R5)
    muvar = R6 / (R1 * W0 * C1 * R5)
    E0 = R6 / (R4 * R5 * R7 * C1 * C2 * (W0 ** 2))
    U1R1f = 1 / (R1 * C1 * 2 * numpy.pi)
    U1R2f = 1 / (R2 * C1 * 2 * numpy.pi)
    U2f = 1 / (R7 * C2 * 2 * numpy.pi)

    print(f"W0 = {round(W0, 2)} rad/s\tf = {round(f,2)} Hz\tmu = {round(mu,2)}\tE0 ={round(E0,2)}"
          f"\tU1R1f = {round(U1R1f,2)} Hz\tU1R2f = {round(U1R2f, 2)} Hz\tU2f = {round(U2f,2)} Hz")
    print(f"\n\'{round(W0, 2)}\t\'{round(f, 2)}\t\'{round(muvar, 2)}\t\'{round(E0, 2)}"
          f"\t\'{round(U1R1f, 2)}\t\'{round(U1R2f, 2)}\t\'{round(U2f, 2)}")
    print(f"'{R1}\t'{R2}\t'{R3}\t'{R4}\t'{R5}\t'{R6}\t'{R7}\t'{C1}\t'{C2}\t")