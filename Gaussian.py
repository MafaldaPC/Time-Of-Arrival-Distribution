import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
            x = xVar - self.P[idx]*tVar/self.M[idx] 
            
        wavepckt = 1/( np.pi**(1/4)*np.sqrt( self.S[idx] + 1j*t/(self.S[idx]*self.M[idx]) ) ) 
        wavepckt *= np.exp( -1*( (x - self.X[idx] - self.P[idx]*t/self.M[idx])**2 )/\
                          ( (2*self.S[idx]**2)*(1 + 1j*t/( self.M[idx]*self.S[idx]**2 ) ) ) )
        wavepckt *= np.exp( 1j*self.P[idx]*( x - self.X[idx] - self.P[idx]*t/(2*self.M[idx]) ) )
        return wavepckt

  
    def condarrival(self, tVar, xVar, x1, x2, click, numclick, idx):
        """ Evaluate single Gaussian wave packet conditioned on being measured at this specific time and not before.
        Parameters:
        tVar : float or array-like
            Time variable.
        xVar : float or array-like
            Position variable.
        idx : int
            Packet index.
        x1 : int
            Left edge of detector.
        x2: int 
            Right edge of detector.
        click: float
            Time spent between two consecutive clicks.
        numclick: float
            Number of time points between two consecutive clicks
        Returns:
        finalwave: Complex or np.ndarray
            Value of Gaussian packet at every time point tVar conditioned on being on [x1,x2], not having been there at prior times
         """

        def Operator( p, click, idx):
            """ Computes the time evolution of one click with momentum p.
            x : float or array-like
                Position variable.
            xp : float or array-like
                Prior position of the packet.
            Returns:
                Value of the Unitary Operator with (x, click, xp).
            """
            return np.exp(-1j*click*p**2/(2*self.M[idx]))

        def Wavepacket( t, x, idx):
            """ Evaluate single Gaussian wave packet.
            t : float or array-like
                Time variable.
            x : float or array-like
                Position variable.
            Returns:
              wavepckt : complex or np.ndarray
            Value of the Gaussian packet at (t, x).
            """
            return (1/( np.pi**(1/4)*np.sqrt( self.S[idx] + 1j*t/(self.S[idx]*self.M[idx]) ) ) *\
                    np.exp( -1*( (x - self.X[idx] - self.P[idx]*t/self.M[idx])**2 )/\
                           ( (2*self.S[idx]**2)*(1 + 1j*t/( self.M[idx]*self.S[idx]**2 ) ) ) )*\
                    np.exp( 1j*self.P[idx]*(x- self.X[idx] - self.P[idx]*t/(2*self.M[idx]) ) ))

        t = np.array(tVar)
        x = [np.array(xVar)]*len(idx) #List of positions for each packet
        dx = x[0][1]-x[0][0]
        numPoints = len(x[0])
        x1 = [x1]*len(idx)
        x2 = [x2]*len(idx)
        p = 2*np.pi*np.fft.fftfreq(numPoints, d=dx) #momentum
        if self.nospread:
            for k in idx:
                x[k] = x[k] - self.P[k]*t/self.M[k] 
                x1[k] = x1[k] - self.P[k]*t/self.M[k]
                x2[k] = x2[k] - self.P[k]*t/self.M[k]
            t = np.zeros_like(t)
            
        finalwave = []

        wave= []
        Propagator = []
        for k in idx:
            wave.append(Wavepacket(t,x[k],k))
            Propagator.append(Operator(p,click,k))
        
        mask=[]
        for k in idx:
            maskex = [] # One mask for each packet
            for i in range(numPoints):
                if (x[k][i]>=x1[k]) and (x[k][i]<=x2[k]):
                    maskex.append(i)
            mask.append(maskex)

        for i in range(numPoints):
            if (i%numclick)==0: #If the detector clicks
                prob = 0
                for j in range(numPoints):
                    aux = 0
                    for k in idx:
                        if j in mask[k]:
                            aux += wave[k][j] / np.sqrt(len(mask[k])) # Summing all the packets at this point and dividing it by the squareroot of the number of packets
                            if self.conditioned: #If there is conditioning on non-arrival
                                wave[k][j] = 0 
                    prob += abs(aux)**2 *dx #Probability of the packets being inside of the detector's region
                finalwave.append(prob)

                for k in idx:
                    if self.conditioned:
                        norm = np.sum(np.abs(wave[k])**2) * dx
                        wave[k] = wave[k] / np.sqrt(norm)
                    wave[k] = (np.fft.ifft( Propagator[k] * np.fft.fft( wave[k] ) ))
                    norm = np.sum(np.abs(wave[k])**2)*dx
                    wave[k] = wave[k] / np.sqrt(norm)


        return(finalwave)


    def condsuperposition(self, tVar, xVar, xDtc, click, numclick):
        """ Evaluate superposition of Conditioned Gaussian wave packets.
        Parameters:
        tVar : float or array-like
            Time variable.
        xVar: float or array-like
            Position variable.
        xDtc: float or array-like
            Detector Position.
        numclick: float
            Number of time points between two consecutive clicks
        click: float
            Time spent between two consecutive clicks.
        Returns:
        trainpckt : complex or np.ndarray
            Value of the superposition at (tVar, xVar).
        """
        idx = list(range(self.N))
        trainpckt = self.condarrival(tVar, xVar, xDtc[0], xDtc[1], click, numclick, idx) 
        return trainpckt/np.sqrt(self.N)


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
        for idx in range(self.N): 
            trainpckt += self.packet(tVar, xVar, idx) 
        return trainpckt

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
        densityDtc = np.abs(self.superposition(t, xDtc))**2

        plt.rcParams.update({'font.size': 12})
        plt.rcParams["font.family"] = "serif"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi = 400,\
                                            constrained_layout = True)
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


    def visualizec(self, numPoints, tLim, xLim, xDtc, f, click):
        """ Visualize conditioned wave function with density and fixed-position plotw.
        Parameters:
        numPoints : int
            Number of points in each dimension (x and t).
        tLim : tuple or array-like
            Temporal domain.
        xLim : tuple or array-like
            Spatial domain.
        xDtc : tuple or array-like
            Detector position
        f : int
            Number of Clicks under consideration.
        click : float
            Time spent between two consecutive clicks.
        """
        t = np.linspace(tLim[0], tLim[1], numPoints)
        x = np.linspace(xLim[0], xLim[1], numPoints)
        T, X = np.meshgrid(t, x)



        density = np.abs(self.condsuperposition(T, X, xDtc, f, click))**2 # conditioned wave
        densityDtc = np.abs( sp.integrate.quad(lambda y: self.superposition(t, y), xDtc[0], xDtc[1]))**2 

        plt.rcParams.update({'font.size': 12})
        plt.rcParams["font.family"] = "serif"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi = 400,\
                                            constrained_layout = True) 
        pcm = ax1.pcolormesh(T, X, density, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax = ax1, label="Density")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Position")
        ax1.title.set_text("Density")
        ax2.plot(t, densityDtc, color = 'tab:blue', marker = '.', linestyle = '-',\
                 markersize = 1, linewidth = 0.5)
        ax2.set_xlabel("Time")
        ax2.title.set_text(f"Density at x = [{xDtc[0]},{xDtc[1]}]")
        
        plt.show()
