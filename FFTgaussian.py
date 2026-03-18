import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math


class GaussianTrain():
    """ Build, manage and visualize a superposition of Gaussian wave packets. """
    def __init__(self, X, P, S, M, nospread = False, conditioned = False):
        """ Initialize Gaussian wave packets.
        Parameters:
        X : array-like
            Initial positions of the packets.
        P : array-like
            Initial momenta of the packets.
        S : array-like
            Standard deviations of the packets.
        M : array-like
            Masses of the packets.
        nospread : bool, optional
            Disables packet spreading (default: False).
        """
        self.X = np.array(X, dtype = float) #transformar num array de np
        self.P = np.array(P, dtype = float)
        self.S = np.array(S, dtype = float)
        self.M = np.array(M, dtype = float)
        self.N = np.size(self.X) #número de packets que o wave train tem
        self.nospread = nospread
        self.conditioned = conditioned

    def packet(self, tVar, xVar, idx):
        """ Evaluate single Gaussian wave packet.
        Parameters:
        tVar : float or array-like
            Time variable.
        xVar : float or array-like
            Position variable.
        idx : int
            Packet index.
        Returns:
        wavepckt : complex or np.ndarray
            Value of the Gaussian packet at (tVar, xVar).
        """

        t = np.array(tVar)
        x = np.array(xVar)

        if self.nospread:
            t = np.zeros_like(tVar)
            x = xVar - self.P[idx]*tVar/self.M[idx] #x-p_0*t_0/m = x-x_0
        wavepckt = 1/( np.pi**(1/4)*np.sqrt( self.S[idx] + 1j*t/(self.S[idx]*self.M[idx]) ) ) # Amplitude
        wavepckt *= np.exp( -1*( (x - self.X[idx] - self.P[idx]*t/self.M[idx])**2 )/\
                          ( (2*self.S[idx]**2)*(1 + 1j*t/( self.M[idx]*self.S[idx]**2 ) ) ) )
        wavepckt *= np.exp( 1j*self.P[idx]*( x - self.X[idx] - self.P[idx]*t/(2*self.M[idx]) ) )
        return wavepckt



    def condarrival(self, tVar, xVar, x1, x2, click, numclick, idx): 
        def Wavepacket( t, x, idx):
            return (1/( np.pi**(1/4)*np.sqrt( self.S[idx] + 1j*t/(self.S[idx]*self.M[idx]) ) ) *\
                    np.exp( -1*( (x - self.X[idx] - self.P[idx]*t/self.M[idx])**2 )/\
                          ( (2*self.S[idx]**2)*(1 + 1j*t/( self.M[idx]*self.S[idx]**2 ) ) ) )*\
                    np.exp( 1j*self.P[idx]*(x - self.X[idx] - self.P[idx]*t/(2*self.M[idx]) ) ))
        def Operator( p, click, idx):
            return np.exp(-1j*( click )*p**2/(2*self.M[idx]))


        t = np.array(tVar)
        x = [np.array(xVar).copy()]*len(idx)
        dt = click/numclick
        dx = x[0][1]-x[0][0]
        numPoints = len(x[0])

        p = 2*np.pi*np.fft.fftfreq(2*numPoints, d=dx) # Double points for the padding
            
        
        finalwave = []
        wave = []
        Propagator = []
        for k in idx:
            if self.nospread: #This does not yet work
                t = np.zeros_like(t)
                x[k] = x[k] + self.P[k]*tVar/(self.M[k]) 
                wave.append(Wavepacket(0,x[k],k))
                Propagator.append(Operator(p,click,k))
            else: 
                wave.append(Wavepacket(t,x[k],k))
                Propagator.append(Operator(p,click,k))

        mask=[]

        for k in idx:
            maskex = []
            for i in range(numPoints):
                if (x[k][i]>=x1) and (x[k][i]<=x2):
                    maskex.append(i)
            mask.append(maskex)

        for i in range(numPoints):
            if (i%numclick)==0:
                prob = 0
                for j in range(numPoints):
                    aux = 0
                    for k in idx:
                        if j in mask[k]:
                            aux += wave[k][j] / np.sqrt(len(idx))
                            if self.conditioned:
                                wave[k][j] = 0 
                    prob += abs(aux)**2 * dx
                finalwave.append(prob)

                for k in idx:
                    padded = np.zeros(2 * numPoints, dtype=complex)
                    padded[:numPoints] = wave[k]

                    propagated = np.fft.ifft(Propagator[k] * np.fft.fft(padded))
                    wave[k] = propagated[:numPoints]  # Trim back to original size
        return(finalwave)


    def condsuperposition(self, tVar, xVar, xDtc, click, numclick):
        """ Evaluate superposition of Conditioned Gaussian wave packets.
        Parameters:
        tVar : float or array-like
            Time variable.
        xDtc: float or array-like
            Detector Position.
        f: int
            Number of Clicks under consideration.
        click: float
            Time spent between two consecutive clicks.
        Returns:
        trainpckt : complex or np.ndarray
            Value of the superposition at (tVar, xVar).
        """

        idx = list(range(self.N))
        trainpckt = self.condarrival(tVar, xVar, xDtc[0], xDtc[1], click, numclick, idx) #The waves overlap inside the function
 
        return trainpckt




    def superposition(self, tVar, xVar):
        """ Evaluate superposition of Gaussian wave packets.
        Parameters:
        tVar : float or array-like
            Time variable.
        xVar : float or array-like
            Position variable.
        Returns:
        trainpckt : complex or np.ndarray
            Value of the superposition at (tVar, xVar).
        """
        trainpckt = 0
        for idx in range(self.N): #numero de packets
            trainpckt += self.packet(tVar, xVar, idx) #junta cada packet a funcao de onda
        return trainpckt/np.sqrt(self.N) #normaliza a funcao de onda

    def visualize(self, numPoints, tLim, xLim, xDtc = 0):
        """ Visualize wave function with density and fixed-position plotw.
        Parameters:
        numPoints : int
            Number of points in each dimension (x and t).
        tLim : tuple or array-like
            Temporal domain.
        xLim : tuple or array-like
            Spatial domain.
        xDtc : float
            Detector position
        """
        t = np.linspace(tLim[0], tLim[1], numPoints)
        x = np.linspace(xLim[0], xLim[1], numPoints)
        T, X = np.meshgrid(t, x)
        density = np.abs(self.superposition(T, X))**2 # \rho = |\psi|^2
        densityDtc = np.abs(self.superposition(t, xDtc))**2 #densidade na posicao do detetor

        plt.rcParams.update({'font.size': 12})
        plt.rcParams["font.family"] = "serif"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi = 400,\
                                            constrained_layout = True) # Figuras correspondentes aos dois diferentes gráficos. ax1=wave-train, ax2=chegada ao detetor
        pcm = ax1.pcolormesh(T, X, density, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax = ax1, label="Density")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Position")
        ax1.title.set_text("Density")
        ax2.plot(t, densityDtc, color = 'tab:blue', marker = '.', linestyle = '-',\
                 markersize = 1, linewidth = 0.5)
        ax2.set_xlabel("Time")
        ax2.title.set_text(f"Density at x = {xDtc}")
        
        plt.show()


    
