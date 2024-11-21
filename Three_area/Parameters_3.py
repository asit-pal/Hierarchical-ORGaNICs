import numpy as np


def Norm_matrix(N,std_dev,gaussian_height):
    '''Gaussian Normalization pool within each layers'''
    # std_dev = std_dev
    convMat = np.zeros((N, N))

    for row in range(N):
        # Instead of generating a kernel upfront, we'll compute the Gaussian values with wrap-around in this loop
        for col in range(N):
            # Calculate the distance considering the wrap-around (circular condition)
            circular_distance = min(abs(col - row), N - abs(col - row))
            
            # Gaussian value calculation based on the circular distance
            value = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (circular_distance ** 2) / (std_dev ** 2))
            
            convMat[row, col] = value

        # Normalize the row to make the sum of its elements equal to 1.
        convMat[row] /= np.max(convMat[row])
        convMat[row] = convMat[row]*gaussian_height

    return (convMat + convMat.T)/2.0


def Recurrence_matrix(N, kernel):
    '''Inter-areal recurrent excitation matrix'''
    K = len(kernel)
    convMat = np.zeros((N, N))
    for row in range(N):
        i = 0
        for col in range(int(row - np.floor(K / 2)), int(row + np.ceil(K / 2))):
            #         print(col)
            convMat[row, np.remainder(col + N, N)] = kernel[i]
            i = i + 1
    return convMat


def RescaleEigenvalues(convMat):
    '''Rescale the eigenvalues recurrent excitation matrix'''
    Utmp, Stmp, Vh = np.linalg.svd(convMat)
    Stmp[Stmp > 1] = 1  # Stmp is 1d vector
    Stmp = Stmp * 1.0
    n_convMat = (Utmp @ np.diag(Stmp) @ Vh)
    return n_convMat

def ReceptiveFields(N,theta,M):
    '''Receptive fields matrix'''
    d = int(N / 2)
    pint = d - 1
    if pint < 1:
        pint = 1

    const = np.sqrt(d / N) * np.sqrt((2 ** (2 * pint) * (np.math.factorial(pint)) ** 2)
                                     / (np.math.factorial(2 * pint) * (pint + 1)))
    RFs = np.zeros((N, M))
    for idx in range(N):
        thetaOffset = idx * 2 * np.pi / N
        thetaDiff = (theta - thetaOffset) / 2
        rf = const * np.cos(thetaDiff) ** pint
        RFs[idx, :] = abs(rf)
    return RFs

def get_Wy1y2(N,theta):
    '''Receptive fields matrix'''
    d = int(N / 2)
    pint = d - 1
    if pint < 1:
        pint = 1

    const = np.sqrt(d / N) * np.sqrt((2 ** (2 * pint) * (np.math.factorial(pint)) ** 2)
                                     / (np.math.factorial(2 * pint) * (pint + 1)))
    RFs = np.zeros((N, N))
    for idx in range(N):
        thetaOffset = idx * 2 * np.pi / N
        thetaDiff = (theta - thetaOffset) / 2
        rf = const * np.cos(thetaDiff) ** pint
        RFs[idx, :] = abs(rf)
    return RFs


def setup_parameters(tau=1e-3,kernel=None, N=36, M=None,tauPlus=1*1e-3, **kwargs):
    '''This function sets up the parameters for the ring model.
    Params:
        kernel: kernel for the within area recurrent matrix
        N: number of neurons
        M: number of stimuli
        kwargs: additional arguments'''
    if kernel is None:
        kernel = [0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934,
                  0.41472545, -0.073386624, -0.060944743, 0.02807382] # qmf 9 kernel
        
    # within area recurrent matrix
    W11 = Recurrence_matrix(N, kernel)
    W11 = (W11.T + W11)/2.0 # making symmetric
    W11 = RescaleEigenvalues(W11) # rescaling eigenvalues to one
    W44 = W11 # currently setting same within area recurrent matrix
    W55 = W11 # currently setting same within area recurrent matrix
    # Normalization matrix for each area
    Wn1 = Norm_matrix(N, std_dev=N/2,gaussian_height=1) # Normalization pool within each layers
    Wn4 = Norm_matrix(N, std_dev=N/2,gaussian_height=1) # Normalization pool within each layers
    Wn5 = Norm_matrix(N, std_dev=N/2,gaussian_height=1) # Normalization pool within each layers
    # Inter-areal connectivity matrix
    # theta1 = np.arange(0, 2 * np.pi, 2 * np.pi / 36)
    # Wy1y2 = Wy1y1**2  # For sabilization by feedback plot
    W14 = W11 @ W11 # connectivity matrix 
    W41 = W14.T # Wy2y1 is transpose of Wy1y2
    W45 = W14
    W54 = W41
    W15 = W14
    W51 = W41
    
    if M is None:
        theta = np.arange(0, 2 * np.pi, 2 * np.pi / 360)
        # theta = np.arange(0,   np.pi,  np.pi / 360)
        M = len(theta)
    else:
        theta = np.linspace(0, 2 * np.pi, M)

    # Encoding matrix: Receptive fields are raised cosine
    Wzx = ReceptiveFields(N, theta, M)
    
    pars = {
    'N': N, 'M': M,
    'tauY1': tau, 'tauY4': tau, 'tauY5': tau,
    'tauYPlus1': tauPlus, 'tauYPlus4': tauPlus, 'tauYPlus5': tauPlus,
    'tauP1': tau, 'tauP4': tau, 'tauP5': tau,
    'tauPPlus1': tauPlus, 'tauPPlus4': tauPlus, 'tauPPlus5': tauPlus,
    'tauU1': tau, 'tauU4': tau, 'tauU5': tau,
    'tauUPlus1': tauPlus, 'tauUPlus4': tauPlus, 'tauUPlus5': tauPlus,
    'tauS1': tau, 'tauS4': tau, 'tauS5': tau,
    'tauSPlus1': tauPlus, 'tauSPlus4': tauPlus, 'tauSPlus5': tauPlus,
    'tauBeta1': tau, 'tauBeta4': tau, 'tauBeta5': tau,
    'tauGamma1': tau, 'tauGamma4': tau, 'tauGamma5': tau,
    'tau': tau,'tauPlus': tauPlus,
    'sigma1': 0.1, 'sigma4': 0.1, 'sigma5': 0.1,
    'alpha1': 10.0, 'alpha4': 10.0, 'alpha5': 10.0,
    'dt': tauPlus/3,
    'W11': W11, 'W44': W44, 'W55': W55,
    'W14': W14, 'W41': W41, 'W45': W45, 'W54': W54, 'W15': W15, 'W51': W51,
    'Wn1': Wn1, 'Wn4': Wn4, 'Wn5': Wn5,
    'Wzx': Wzx,
    'beta1': 1.0, 'beta4': 1.0, 'beta5': 1.0,
    'gamma14': 1.0, 'gamma41': 0.0, 'gamma45': 0.0, 'gamma54': 0.0,'gamma51': 0.0,'gamma15': 1.0,
    'b1': 0.5, 'b4': 0.5, 'b5': 0.5,
    'g14': 0.5, 'g41': 0.0, 'g45': 0.0, 'g54': 0.0,'g51': 0.0,'g15': 0.5
}
    
    for k in kwargs:
        pars[k] = kwargs[k]
    
    return pars

