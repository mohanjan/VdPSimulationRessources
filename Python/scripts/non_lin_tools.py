# General
import numpy as np
import math
import time

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm

# File management
import csv
import pickle

# Numerical
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate, hilbert, windows

#=============================================================
#           Constants
#============================================================= 
    
# Custom color map:
grayCm = cm.get_cmap('gray_r',512)
grayCm = ListedColormap(grayCm(np.linspace(0.15, 1, 256)))
grayCm.set_under(color='white')

#=============================================================
#           General functions
#============================================================= 
    
def plotNumbering(axs: list, ABCoffset: int=0, hOffset: float=0.05, vOffset: float=0.95, ):
    # Simply assign a number to each plot for easy referencing
    ABC = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    for i in range(ABCoffset, len(axs)):
        axs[i].text(hOffset, 
                    vOffset, 
                    s=ABC[i], 
                    horizontalalignment='left', 
                    verticalalignment='top', 
                    transform=axs[i].transAxes, 
                    fontweight='semibold', 
                    fontsize='large',
                    bbox=dict(facecolor='white', edgecolor='black')
                    )
    return

def genSegmentedCmap(ax, low: float, high: float, splits: float, cmap):
    # Generate a segmented colormap for use with boxplot
    # and attach it to the ax passed.
    if cmap == None: cmap = plt.cm.viridis
    bounds = np.linspace(low, high, splits)
    norm = BoundaryNorm(bounds, cmap.N)
    cb = plt.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    return cb

def findCorrelation(x: list, y: list, delta_t: float):
    # Finds the correlation between two distributions
    # Returns two values: One is the max integral and the other is 
    # the frequency shift where the maximum integral was reached. 
    if (len(x) != len(y)):
        print('The two spectrums to be compared are not the same size')
        return None
    corr_vector = correlate(in1=x, in2=y, mode='full', method='direct')
    return corr_vector

#=============================================================
#           Classes
#============================================================= 
    
# Simulate a non linear model. Outputs a raw JSON file
class NonLinMdl:
    #=============================================================
    #           Setup and initialization
    #============================================================= 
    def __init__(self, mdlName: str='VDP', N: int=100, fMin: float=1.0, fMax: float=6.0, P: list=[0,0,0,0], drvP: list=None, tSim: float=50, sFreq: float=1000, additive_noise: bool=False):
        
        # Driving parameters:
        if drvP == None: 
            self.drvP=[['impulse', 1, 0, 0.05]] 
        else: 
            self.drvP = drvP
            
        # set model parameters
        self.P = P 
        self.model = mdlName
        if self.model == 'VDP':
            # Vand Der Pol
            self.dampingFunc = self.__van_der_pol
            self.EQNUM = 2
            self.d, self.k, self.mu, self.v = P
        elif self.model == 'vilfan_duke':
            # Vilfan Duke
            self.dampingFunc = self.__vilfan_duke
            self.EQNUM = 2
            self.d, self.k, self.eps, self.mu, self.v = P
        elif self.model == 'rossler':
            # rössler
            self.dampingFunc = self.__rossler
            self.EQNUM = 3
            self.d, self.k, self.a, self.b, self.c, self.v = P
        else:
            print('Model not found')
        
        # Set system parameters        
        self.N = N # Number of oscilators
        self.fMin = fMin
        self.fMax = fMax
        omega_min = 2*np.pi*fMin
        omega_max = 2*np.pi*fMax
        self.omegaVector = np.linspace(omega_min, omega_max, N) # Generate a linear gradient of eigen frequencies
        
        if not additive_noise: 
            noise = (1 - 2*self.v*(np.random.uniform(size=self.N) - 0.5))
            self.omegaVector *= noise # Apply random noise to gradient 
        else:
            noise = np.random.uniform(low = -self.v, high= self.v, size=self.N)
            self.omegaVector += noise
        
        # Time parameters
        self.tSim = tSim # How long to run the simulation
        self.sFreq = sFreq # Sampling frequency
        self.tVector = np.linspace(0, self.tSim, int(self.sFreq*self.tSim)) # Time vector
        
        # Set driving function
        self.driveFunc = self.__multiDrive 
        
        # Save parameters
        mdlStr = mdlName
        genStr = '_N={}_f=[{:g}_{:g}]'.format(N, fMin, fMax)
        paramStr = '_P=['
        for i in P: paramStr += '{:g}_'.format(i)
        tStr = ']_Tsim={:g}_Fs={:g}'.format(tSim, sFreq)
        flatDrv = [item for sublist in self.drvP for item in sublist]
        drvStr = '_Drv=['
        for i in flatDrv: drvStr += '{}_'.format(i)
        drvStr += ']'
        self.fBase = mdlStr + genStr + paramStr + tStr + drvStr 
                
    #=============================================================
    #           Oscillator models
    #============================================================= 
    
    def __van_der_pol(self, state, omega: float):
        # Scaling has been applied to this function ([vel, -omega**2*pos+self.mu*(1-pos**2)*vel])
        pos = state[0::self.EQNUM]
        vel = state[1::self.EQNUM]
        out_state=np.zeros_like(state)
        out_state[0::self.EQNUM] = omega*vel
        out_state[1::self.EQNUM] = -omega*pos+omega*self.mu*(1-pos**2)*vel
        return(out_state)
    
    def __vilfan_duke(self, z: complex, omega: float):
        # Complex Vilfan Duke
        func = (omega*1j + self.eps)*z - self.mu*abs(z)**2*z
        return(func)
    
    def __rossler(self, state, omega: float):
        # Rossler oscillator with three equations
        pos = state[0::self.EQNUM]
        vel = state[1::self.EQNUM]
        acc = state[2::self.EQNUM]
        out_state = np.zeros_like(state)
        out_state[0::self.EQNUM] = -omega*vel - acc
        out_state[1::self.EQNUM] = omega*pos + self.a*vel
        out_state[2::self.EQNUM] = self.b + acc*(pos - self.c)
        return out_state
    
    def __couplingFunc(self, pos_diff: list, vel_diff: list):
        #TODO: vectors of d and k not implemented, it's assumed they are identical
        #Vectors of d and k will be useful when implemnting the option to have no wall
        #Dissapative coupling
        func = self.d*(vel_diff[2]-vel_diff[1]) + self.d*(vel_diff[0]-vel_diff[1]) 
        
        # Add reactive coupling 
        func += self.k*(pos_diff[2]-pos_diff[1]) + self.k*(pos_diff[0]-pos_diff[1])
        return (func)
    
    def __complexCouplingFunc(self, z_diff: list):
        # To be used with the Vilfan Duke model
        # TODO: I accidently mixed around d and k here. 
        # Real should be d, imag should be k
        func = (self.k+self.d*1j)*(z_diff[2]+z_diff[0]-2*z_diff[1])
        return func
        
    #=============================================================
    #           Driving functions
    #============================================================= 
    
    def __multiDrive(self, drvP: list, t: float):
        # Returns all contributions from all n different drvFuncs
        n = len(drvP)
        funcSum = 0.0
        for i in range(n):
            drvStr = drvP[i][0]
            if drvStr == 'sin':
                a, f, dStart, dEnd = drvP[i][1:]
                funcSum += self.__sinDrive(a, f, dStart, dEnd, t)
                
            elif drvStr == 'sweep':
                a, f0, f1, tStart, tEnd = drvP[i][1:]     
                funcSum += self.__sweepDrive(a, f0, f1, tStart, tEnd, t)
                
            elif drvStr == 'noise':
                mean, sd, dStart, dEnd = drvP[i][1:]
                funcSum += self.__noiseDrive(mean, sd, dStart, dEnd, t)
                
            elif drvStr == 'impulse':
                a, pStart, pWidth = drvP[i][1:]
                funcSum += self.__impulseDrive(a, pStart, pWidth, t)
                
            elif drvStr == 'double':
                a, f1, f2, dStart, dEnd = drvP[i][1:]
                funcSum += self.__modDrive(a, f1, f2, dStart, dEnd, t)
            
            else:
                print('Could not read parameter {}'.format(drvStr))
        
        return (funcSum)
        
    def __sinDrive(self, a: float, f: float, dStart: float, dEnd: float, t: float):
        # Sinusoid for driving oscillations
        if dStart <= t < dEnd:
            return a*np.sin(2.0*np.pi*f*t) 
        return 0
    
    def __sweepDrive(self, a: float, f0: float, f1: float, tStart: float, tEnd: float, t: float):
        if tStart <= t < tStart+tEnd:
            return a*np.sin( 2*np.pi*(f0*t + (f1-f0)/(2*(tEnd-tStart))*(t-tStart)**2 )  )
        return 0    
    
    def __noiseDrive(self, mean: float, sd: float, dStart: float, dEnd: float, t):
        # Random noise for driving oscillations
        if dStart <= t < dEnd:
            return np.random.normal(mean, sd, 1)
        return 0
    
    def __impulseDrive(self, a: float, pStart: float, pWidth: float, t: float):
        # Impulse function for starting oscilations
        if pStart <= t < pStart+pWidth: 
            return a 
        return 0
    
    def __modDrive(self, a: float, f1: float, f2: float, dStart: float, dEnd: float, t: float):
        # For Bepp's experiment
        if dStart <= t < dEnd:
            return (1+np.sin(2*np.pi*f1*t)*a*np.sin(2*np.pi*f2*t))
        return 0
    
    #=============================================================
    #           Setting up system and solving
    #=============================================================
    
    def __sysEq(self, t, state):
        # Initialize and add model
        diff_state = self.dampingFunc(state, self.omegaVector)
        # Add coupling:
        for i in range(0, self.N, 1):
            pos = state[self.EQNUM*i]      # x or x1 
            vel = state[self.EQNUM*i+1]    # y or x2
            
            # Add coupling function
            pos_diff = np.zeros(3) 
            vel_diff = np.zeros(3)
            
            pos_diff[1] = pos
            vel_diff[1] = vel
            
            if i != 0: #If previous node is NOT a wall
                pos_diff[0] = state[self.EQNUM*(i-1)]
                vel_diff[0] = state[self.EQNUM*(i-1)+1]
            
            if i < self.N-1: #If next node is NOT a wall
                pos_diff[2] = state[self.EQNUM*(i+1)]
                vel_diff[2] = state[self.EQNUM*(i+1)+1]
            
            diff_state[self.EQNUM*i+1] += self.__couplingFunc(pos_diff, vel_diff)
            
        # Add drive function
        diff_state[1::self.EQNUM] += self.driveFunc(self.drvP, t)
        
        return diff_state
        
    def __sysEqComplex(self, t, state):
        diff_state = np.zeros_like(state, dtype='complex128')
        # A function for handling the complex Ginzburg Landau oscillator
        # Append all N couples equations.
        for i in range(0, self.N, 1):
            z = state[i]
            
            # Add chosen model
            diff_state[i] = self.__vilfan_duke(z, self.omegaVector[i])
            
            # Add coupling function
            z_diff = np.zeros(3, dtype='complex128') 
            
            z_diff[1] = z
            
            if i != 0: #If previous node is NOT a wall
                z_diff[0] = state[i-1]
                
            if i < self.N-1: #If next node is NOT a wall
                z_diff[2] = state[i+1]
                
            diff_state[i] += self.__complexCouplingFunc(z_diff)
            
        # Add drive function
        diff_state += self.driveFunc(self.drvP, t)*1j
       
        return diff_state
        
    def solveSystem(self, init_state: np.array=None):
        if init_state == None: 
            if self.model == 'vilfan_duke': init_state=np.zeros(self.N*1, dtype='complex128') # TODO: Special exception for Vilfan Duke
            else: init_state=np.zeros(self.N*self.EQNUM)
        start = time.time()
        
        if self.model=='vilfan_duke':
            solveFunc = self.__sysEqComplex
        else: 
            solveFunc = self.__sysEq
  
        sol = solve_ivp(fun=solveFunc, t_span=(0.0, self.tSim), y0=init_state, t_eval=self.tVector, max_step=0.01, method="RK45")
        self.sol_t = sol.t
        
        if self.model=='vilfan_duke':
            self.sol_y = np.zeros([2*self.N, len(sol.t)])
            self.sol_y[0::2,:] = (sol.y).real
            self.sol_y[1::2,:] = (sol.y).imag
        else:
            self.sol_y = sol.y
            
        end = time.time()
        print('Solved system in {:.2f} seconds'.format(end-start))
        return 
          
    def getSolution(self, tInv: list=[0,None]):
        # Outputs the displacement x and velocity y of the solution.
        t0, t1 = tInv
        t = self.sol_t[t0:t1]
        pos = self.sol_y[0::self.EQNUM,t0:t1]
        vel = self.sol_y[1::self.EQNUM,t0:t1]
        return [t, pos, vel]
    
    #===================================================
    #               Saving and loading data
    #===================================================
    
    def saveSol(self, note: str=''):
        # Save the solution to a file in pickle-format
        # File name:
        fPath = '../data/' + self.fBase + note + '.p'
        saveDict = {
            'mdlName': (self.model),
            'N': self.N,
            'fMin': self.fMin,
            'fMax': self.fMax, 
            'P': self.P,
            'drvP': self.drvP,
            'tSim': self.tSim,
            'sFreq': self.sFreq,
            'omegaVector': self.omegaVector.tolist(), # JSON can't save np array
            'fileName': self.fBase,
            't': list(self.sol_t), 
            'y': self.sol_y.tolist()
        }
        
        f = open( fPath, "wb" )
        pickle.dump(saveDict, f)
        print('Saved system at {}'.format(fPath))
        f.close()
        
    def saveSolcsv(self, note: str=''):
        # saves only the t and y in a csv file
        fPath = '../data/' + self.fBase + note + '.csv'
        f = open(fPath,"w")
        saveData = np.c_[self.sol_t, self.sol_y.transpose()]
        
        # Write data to file
        writer = csv.writer(f)
        writer.writerows(saveData)
        f.close()
        
    def loadSol(self, fName: str):
        # Load a pickle formatted file
        fPath = '../data/' + fName + '.p'
        f =  open( fPath, "rb" )
        json_obj = pickle.load(f)
        
        mdlName = json_obj['mdlName']
        N = json_obj['N']
        fMin = json_obj['fMin']
        fMax = json_obj['fMax']
        P = json_obj['P']
        drvP = json_obj['drvP']
        tSim = json_obj['tSim']
        sFreq = json_obj['sFreq']
        omegaVector = json_obj['omegaVector']
        fileName = json_obj['fileName']
        
        # Let the init function set our variables
        self.__init__(mdlName=mdlName,
                      N=N,
                      fMin=fMin,
                      fMax=fMax,
                      P=P,
                      drvP=drvP,
                      tSim=tSim,
                      sFreq=sFreq)
        
        # The variables set by init we need to overwrite
        self.sol_t = np.array(json_obj['t'])
        self.sol_y = np.array(json_obj['y'])
        self.omegaVector = np.array(omegaVector)
        self.fBase = fileName
        
        print('Loaded system from {}'.format(fPath))
        
        f.close()
        
    #===================================================
    #           Analysis tools
    #===================================================
    
    def findTime(self, chosen_t: float):
        # Find the corrosponding time index for a specified time
        return_t = int(chosen_t*self.sFreq)
        if return_t > len(self.sol_t):
            return_t = len(self.sol_t)
        
        print('Index {} corrosponds to time {}'.format(return_t, self.sol_t[return_t]))
        return return_t
         
    def findEnergy(self, tInv: list):
        # Calculate approximate energy of oscillator within a time interval
        # Returns vector of energy calculated for each N
        t0 = tInv[0]
        t1 = tInv[1]
        pos = self.sol_y[::self.EQNUM, t0:t1]
        Energy = np.zeros(self.N)
        
        for i in range(self.N):
            Energy[i] = (pos[i]**2).sum()
        return Energy
    
    def findSteadyState(self, Tint: float, Tstart: float=0, threshold: float=0.1):
        # Detects when steady state is reached and outputs the corrosponding index
        # Steady state detection starts at Tstart, and then checks in intervals 
        # of Tint, if a steady state has been reached
        indxInt = int(Tint*self.sFreq)
        indxEnd = len(self.sol_t)
        indexStart = int(Tstart*self.sFreq)
        enDiff = np.zeros(self.N)
        crtEn = self.findEnergy([indexStart, indexStart+indxInt])
        
        for i in range(indexStart+indxInt, indxEnd, indxInt):
            prevEn = crtEn
            crtEn = self.findEnergy([i, i+indxInt])
            enDiff = abs(crtEn-prevEn)/prevEn
            print(max(enDiff)) #TODO: debug
            print('------------------')
            if max(enDiff) < threshold:
                print('Steady state found at time {:.2f}'.format(self.sol_t[i]))
                return i
        print('Steady state not found')
        return 0
    
    def genTimeIntervals(self, splits: int, cumulative: bool= False, tStart: float=0, tEnd: float=-1):
        # TODO: deprecated
        # Create a list of time intervals for use in other functions
        intM = np.zeros([splits, 2], dtype=np.uint32)
        indxStart = self.findTime(tStart) 
        if tEnd < 0: 
            indxEnd = len(self.sol_t)-1 # if end is -1, get last index of array
        else:
            indxEnd = self.findTime(tEnd)
        intM[0][0]=indxStart
        intM[-1][-1]=indxEnd
        delta = math.ceil((indxEnd+1-indxStart)/splits)
        for i in range(1,splits):
            intM[i-1][1] = indxStart + i*delta - 1
            intM[i][0] = indxStart + i*delta*(not cumulative) # don't increment indexStart if cumulative
        return intM
    
    def genOverlapIntervals(self, dist: float, width: float=None, tStart: float=0, tEnd: float=None):
        # Create multiple overlapping time intervals from width and distance
        # Also returns the array with centers of windows
        # All parameters are given in seconds, not bins
        if tEnd==None: tEnd = self.sol_t[-1]
        centers = np.arange(tStart, tEnd, dist)
        intM = np.zeros([len(centers), 2], dtype=np.uint32)
        for i in range(len(centers)):
            # Clamp the intervals to avoid overflow
            intM[i,0] = max(0, int( (centers[i]-width)*self.sFreq))
            intM[i,1] = min(len(self.sol_t)-1, int( (centers[i]+width)*self.sFreq))
        return intM, centers
           
    def zeroCrossings(self, t: list, y: list):
        # TODO: Note that it can't detect 0 - might cause bugs in the furture
        # Returns all periods derived from zero crossings of the passed vector.
        # Locate all crossings:
        yloc = np.zeros(len(y), dtype=bool)
        for i in range(1, len(y)):
            if abs(np.sign(y[i])-np.sign(y[i-1])) > 1: yloc[i] = True
        # Use locations of crossings in t vector, and find the differences
        zCrossVector = 2*np.diff(t[np.where(yloc)])
        return zCrossVector
    
    def ZCEstimate(self, tInv=[0,None]):
        # Returns the ZC estimate matrix of shape (N, maxLength) in the time
        # interval tInv
        # get y and t from solution:
        t, y = self.getSolution(tInv)[0:2]
        # First, save all ZC frequencies in jagged list
        N = y.shape[0]
        ZClist = [None]*N # Can't create jagged np array, have to convert from list
        maxLen = 0
        for i in range(N):
            ZClist[i] = self.zeroCrossings(t=t, y=y[i,:])
            if len(ZClist[i]) > maxLen: maxLen = len(ZClist[i])
        # Re-arrange to numpy array:
        ZCmatrix = np.empty((N,maxLen))
        ZCmatrix[:] = np.nan # nan used to represent jagged list
        for i in range(N):
            ZCmatrix[i, 0:len(ZClist[i])] = ZClist[i]   
        # Also, get the frequency means:
        ZCmeans = np.nanmean(ZCmatrix, axis=1)**-1
        return ZCmatrix, ZCmeans
        
    def instFreqEstimate(self, tInv=[0,None], hilbertEstimate=True):
        # Input: Matrix of amplitudes of size (N, t)
        # Output Matrix of size (N, t-1) of estimated freqency
        # If hilbert=true:
        # Estimates the instantenous frequency, from the hilbert transform
        # Else: estimates it from x and y
        x, y = self.getSolution(tInv)[1:3]
        N = x.shape[0]
        instFreqM = np.zeros([N, x.shape[1]-1])
        for i in range(N):
            if hilbertEstimate:
                # Phase from Hilbert:
                phase = np.unwrap(np.angle(hilbert(x[i,:])))
            else: 
                # Otherwise, phase from x and y
                phase = np.unwrap(np.angle( (x[i,:]-y[i,:]*1j) ))
            # Estimate derivative:
            instFreqM[i,:] = np.diff(phase) / (2.0*np.pi) * self.sFreq
        return instFreqM
    
    def zeroCrossingQuartiles(self, y: list, p: list=[0.25, 0.5, 0.75]):
        # Get quartiles (optional: percentiles) for zero crossings periods
        y.sort()
        n = len(y)
        Q = np.zeros_like(p)
        for i in range(len(p)):
            pn = p[i]*(n-1)
            
            if pn==0:
                Q[i] = min(y)
            elif pn==n:
                Q[i] = max(y)
            elif pn.is_integer():
                Q[i] = (y[int(pn)-1]+y[int(pn)])/2
            else:
                Q[i] = y[math.ceil(pn)-1]
        return Q
        
    def genPhaseTrajectory(self, N: int, tInv: list=[0,-1]):
        # Returns the position, velocity and time of a chosen oscillator in a chosen interval
        tStart, tEnd = tInv
        n = N -1 # Make sure that oscillator starts in 0
        pos = self.sol_y[n*self.EQNUM,  tStart:tEnd]
        vel = self.sol_y[n*self.EQNUM+1,tStart:tEnd]
        t = self.sol_t[tStart:tEnd]
        return(pos, vel, t)
                 
    def gen2DHist(self, y: np.array, minValue: float, maxValue: float, bins: int):
        # Generates histogram data for imshow(). Takes a matrix as an input, 
        # where the first dimension is N, second is data
        # Initialize arrays and matrices
        N = y.shape[0]
        imMatrix = np.zeros((bins,N))
        
        for i in range(N):
            imMatrix[:,i] = np.histogram(a=y[i,:], bins=bins, range=[minValue, maxValue])[0]
            #Normalize each column:
            if max(imMatrix[:,i]) != 0:
                   imMatrix[:,i] = imMatrix[:,i] / max(imMatrix[:,i])
        return imMatrix
        
    def genFourierSpectrum(self, y: np.ndarray, fs: float, windowing: bool=True):
        # Fourie transform of a signal
        # If we want to window our signal, use Hann:
        if windowing == True: y = y*windows.hann(len(y))
        aVector = rfft(y)
        fVector = rfftfreq(n=len(y), d=1/fs)
        return [aVector, fVector]
    
    def genFourierMatrix(self, tInv: list=[0,None], windowing: bool=True):
        # Gets the Fourie spectras for all oscillators in Nlist and arranges them
        # in a matrix of shape [N, len(spectrum)]:
        pos = self.getSolution(tInv=tInv)[1]
        fVector = rfftfreq(n=pos.shape[1], d=1/self.sFreq)
        aMatrix = np.zeros([self.N, len(fVector)], dtype='complex128')
        for i in range(self.N):
            aMatrix[i,:] = self.genFourierSpectrum(y=pos[i], fs=self.sFreq, windowing=windowing)[0]
        return aMatrix, fVector
    
    def findMeanFreq(self, fVector: np.ndarray, aMatrix: np.ndarray):
        # Get the mean frequency from a matrix of spectras of shape [N, spectra length]
        # Output an array of mean frequencies for each oscillator in matrix
        n = aMatrix.shape[0]
        meanFreq = np.zeros(n)
        magMatrix = abs(aMatrix) #convert to magnitudes
        # Nomalize by sum of spectrum:
        for i in range(n):
            meanFreq[i] = sum( (magMatrix[i,:]/sum(magMatrix[i,:]))*fVector )
        return meanFreq
    
    def findPeakFreq(self, fVector: np.ndarray, aMatrix: np.ndarray):
        # Get the peak frequency and index from a matrix of spectras of 
        # shape [N, spectra length]
        # Output an array of peak frequencies and index for each oscillator in matrix
        n = aMatrix.shape[0]
        peakFreq = np.zeros(n)
        peakIndex = np.zeros(n, dtype='int64')
        magMatrix = abs(aMatrix) #convert to magnitudes
        for i in range(n):
            # Two indexes is because the function outputs a tuple with an array.
            # We want the first element of the array
            peakIndex[i] = np.where( magMatrix[i,:]/max(magMatrix[i,:]) == 1.0)[0][0] 
            peakFreq[i] = fVector[peakIndex[i]]
        return peakFreq, peakIndex
        
    def findSyncStatus(self, fMeans: np.ndarray, min_clustering: int=4):
        # input: matrix of mean frequency for each oscillator n (that can be ZC
        # means, peak frequency etc.)
        # Returns sync vector, coded according to how the clusters fit together
        # Computes which oscillators are clustered together.
        # 0 = no clustering (below threshold of min_culstering), 
        # any other number = clustering with other corrosponding numbered oscillators. 
        sync_vector = np.zeros(self.N, dtype='int8')
        # First, find the ZC means in interval:
        means_vector = fMeans
        min_w = min(means_vector)
        max_w = max(means_vector)
        # calculate stepsize:
        step_size = np.diff(means_vector)
        # Find and code clusters:      
        code = 1
        ii = 0
        while ii < (self.N-1):
            j = (self.N-1)
            while j > ii:
                max_step = max(step_size[ii:j])
                if (max_step < (max_w-min_w)/(2*self.N)):
                    # a cluster has been found! Code all oscillators as being 
                    # part of a cluster:
                    print('Cluster found between n={} and n={}'.format(ii, j))
                    # Check if cluster is large enough:
                    if ((j-ii+1) >= min_clustering):
                        sync_vector[ii:j+1] = code
                        code += 1    
                    else: print('Cluster too small. Ignoring it')
                    break
                else:
                    j -=1
            # Update the iteration of the loop, to start from end of 
            # previous cluster
            ii = j+1
        return sync_vector
        
    def genSyncMetrics(self, sync_vector: list):
        # Very simple function, that simply returns metrics suchs as 
        # number of clusters, mean cluster size and sync percantage:
        # first, number of clusters:
        numClus = max(sync_vector)
        # Mean cluster size:
        sizes = np.zeros(numClus)
        for i in range(1, numClus+1):
            sizes[i-1] = sum(sync_vector == i)
        # Get mean:
        meanClus = sum(sizes)/numClus
        # Get prcentage:
        percClus = sum(sync_vector > 0)/len(sync_vector)
        return numClus, meanClus, percClus
                
    #===================================================
    #           System Plotting tools
    #===================================================
    
    def saveFig(self, pName: str=None, pFormat: str='eps'):
        # Saves the figure with the appropiate name and in the correct folder
        if pName == None: pName = self.fBase + '_fig' 
        fPath = "../figures/"
        plt.savefig(f'{fPath}{pName}.{pFormat}', format=pFormat)
        print('Saved figure {} at location {}'.format(pName,fPath))
        return
        
    def plotZCMean(self, ax, means, Nlist: list=None, color: str='r', marker: str='+', size: float=30, linewidth: float=1):
        # Plots all zc means found in Nlist
        if Nlist == None: Nlist = np.linspace(1,self.N, self.N, dtype='int16')
        meansplot = means[(Nlist-1)]        
        ax.scatter(Nlist, meansplot, color=color, marker=marker, s=size, linewidth=linewidth)
        return 
    
    def plotZCBoxplot(self, ax, Nlist: list=None, tInv=[0, -1], color='r'):
        # Plots the boxplot of the ZC period distribution in the timeinterval tInv.
        tStart, tEnd = tInv
        if Nlist == None: Nlist = list(range(1,self.N+1))
        Nlen = len(Nlist)
        zc = [None]*Nlen
        p = [0.0, 0.25, 0.5, 0.75, 1.0] # Quartiles for boxplot
        Q = np.zeros([Nlen, len(p)])
        lineStyles = ['-.', '--', '-', '--', '-.'] # Linestyle for each quartile
        for i in range(Nlen):
            n = (Nlist[i]-1) # Oscillators numbered from 1, convert to 0
            zc[i] = self.zeroCrossings(self.sol_t[tStart:tEnd], self.sol_y[n*self.EQNUM,tStart:tEnd])
            Q[i] = self.zeroCrossingQuartiles(zc[i], p=p)
        # then plot:
        for j in range(len(p)):
            ax.plot(Nlist, Q[:,j], color=color, linestyle=lineStyles[j])
        return
            
    def plotFreqColorSpectrum(self, ax, a: np.ndarray, f: np.ndarray, minValue: float=0, maxValue: float=7):
        # a: matrix of frequencies. f: frequency vector. 
        # Plots the frequency spectrum for all N oscillators
        # TODO: Maybe fix limits, so we don't start in 0?
        aMag = abs(a) # get amplitude
        # Normalize each spectra relative to max value:
        for i in range(self.N):
            aMag[i, :] = aMag[i,:]/max(aMag[i,:])
        aMag = aMag.transpose()
        im = ax.pcolormesh(np.arange(self.N), f, aMag, shading='nearest')
        ax.set_ylim([minValue, maxValue])
        return im
        
    def plotDistribution(self, ax, y, minValue: float=None, maxValue: float=None, pltOffset: float=0.5, bins: int=75, cmap=None, fCutoff: float=0.001):
        # Plots the image with the correct configurations
        # Returns the im for use in a colorbar
        if minValue == None: minValue=self.fMin - pltOffset
        if maxValue == None: maxValue=self.fMax + pltOffset
        if cmap == None: cmap = grayCm
        # Generate image
        imMatrix = self.gen2DHist(y, minValue=minValue, maxValue=maxValue, bins=bins)
        # Plot image
        im = ax.imshow(X=imMatrix, extent=(0.5, imMatrix.shape[1]+0.5, minValue, maxValue), cmap=cmap,interpolation='none',aspect='auto', origin='lower', vmin=fCutoff)
        # Axis settings 
        # ax.grid()
        xticksize = 20 if self.N > 50 else 10
        ax.set_xticks(np.arange(0, self.N+xticksize, xticksize))
        ax.set_xticks(np.arange(1, self.N+1), minor=True)
        #ax.set_yticks(np.arange(0, maxValue, 0.5), minor=True)
        ax.tick_params(axis="x",which='both', direction="in")
        ax.tick_params(axis="y",which='both', direction="in")
        ax.set_xlim([1.5, self.N + 0.5])
        
        ax.set_ylim([minValue,maxValue])
        return im
    
    def plotTempAmplitudes(self, ax, tInv: list=[0, None], vmin=None, vmax=None):
        # Plot the amplitudes of each oscillator over time.
        t, pos = self.getSolution(tInv)[0:2]
        pos = pos.transpose()
        im = ax.pcolormesh(np.arange(self.N), t, pos, shading='nearest', cmap='gray_r', vmin=vmin, vmax=vmax, rasterized=True)
        # Setup ticks, for x-axis:
        ax.set_xticks(np.arange(19, self.N, 20))
        ax.set_xticks(np.arange(0, self.N, 1), minor=True)
        ax.set_xticklabels(np.arange(20, self.N+1, 20))
        ax.tick_params(axis="x",which='both', direction="in", bottom=True, top=True, labelbottom=True, labeltop=False)
        ax.tick_params(axis="y",which='both', direction="in")
        return im
    
    def plotInstantFrequency(self, ax, tInv=[0, None], vmin=None, vmax=None):
        # Plot the instantanous frequency for each oscillator.
        # TODO: perhaps combine this function with the one above?
        t = self.getSolution(tInv=tInv)[0] 
        fm = self.instFreqEstimate(tInv=tInv)
        fm = fm.transpose()
        im = ax.pcolormesh(np.arange(self.N), t[:-1], fm, shading='nearest', cmap='gray_r', vmin=vmin, vmax=vmax, rasterized=True)
        # Setup ticks, for x-axis
        ax.set_xticks(np.arange(19, self.N, 20))
        ax.set_xticks(np.arange(0, self.N, 1), minor=True)
        ax.set_xticklabels(np.arange(20, self.N+1, 20))
        ax.tick_params(axis="x",which='both', direction="in", bottom=True, top=True, labelbottom=True, labeltop=False)
        ax.tick_params(axis="y",which='both', direction="in")
        return im 
    