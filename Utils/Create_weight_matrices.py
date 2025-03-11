import numpy as np
import math

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

    const = np.sqrt(d / N) * np.sqrt((2 ** (2 * pint) * (math.factorial(pint)) ** 2)
                                     / (math.factorial(2 * pint) * (pint + 1)))
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

    const = np.sqrt(d / N) * np.sqrt((2 ** (2 * pint) * (math.factorial(pint)) ** 2)
                                     / (math.factorial(2 * pint) * (pint + 1)))
    RFs = np.zeros((N, N))
    for idx in range(N):
        thetaOffset = idx * 2 * np.pi / N
        thetaDiff = (theta - thetaOffset) / 2
        rf = const * np.cos(thetaDiff) ** pint
        RFs[idx, :] = abs(rf)
    return RFs


def setup_parameters(config, tau=1e-3, kernel=None, N=36, M=None, tauPlus=1*1e-3, **kwargs):
    '''This function sets up the parameters for the ring model.
    Params:
        config: configuration dictionary from YAML
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
    # Wn1 = Norm_matrix(N, std_dev=N/2,gaussian_height=1) # Normalization pool within each layers
    # Wn2 = Norm_matrix(N, std_dev=N/2,gaussian_height=1) # Normalization pool within each layers
    Wn1 = np.ones((N,N))
    Wn2 = np.ones((N,N))
    # Wn1 = (Wn1 + Wn1.T)/2.0 # making symmetric
    # Wn2 = (Wn2 + Wn2.T)/2.0 # making symmetric
    W44 = W11 # currently setting same within area recurrent matrix
    # theta1 = np.arange(0, 2 * np.pi, 2 * np.pi / 36)
    # Wy1y2 = Wy1y1**2  # For sabilization by feedback plot
    W14 = W11 @ W11 # connectivity matrix 
    # W14 = W11
    W41 = W14.T # Wy2y1 is transpose of Wy1y2
    W45 = W14
    
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
        'tauY1': tau, 'tauY4': tau,
        'tauYPlus1': tauPlus, 'tauYPlus4': tauPlus,
        'tauP1': tau, 'tauP4': tau,
        'tauPPlus1': tauPlus, 'tauPPlus4': tauPlus,
        'tauU1': tau, 'tauU4': tau,
        'tauUPlus1': tauPlus, 'tauUPlus4': tauPlus,
        'tauS1': tau, 'tauS4': tau,
        'tauSPlus1': tauPlus, 'tauSPlus4': tauPlus,
        'tauBeta1': tau, 'tauBeta4': tau,
        'tauGamma1': tau, 'tauGamma4': tau,
        'tau': tau, 'tauPlus': tauPlus,
        'tauF1': 1*tau, 'tauF4': 1*tau,
        'sigma1': config['model_params']['sigma1'], 'sigma4': config['model_params']['sigma4'],
        'dt': tauPlus/3,
        'W11': W11, 'W44': W44,
        'W14': W14*0.7, 'W41': W41*1.0, 'W45': W45,
        'Wn1': Wn1, 'Wn4': Wn2,
        'Wzx': Wzx*0.7,
        # Load parameters from config
        'alpha1': config['model_params']['alpha1'],
        'alpha4': config['model_params']['alpha4'],
        'beta1': config['model_params']['beta1'],
        'beta4': config['model_params']['beta4'],
        'gamma1': config['model_params']['gamma1'],
        'gamma4': config['model_params']['gamma4'],
        'b1': config['model_params']['b1'],
        'b4': config['model_params']['b4'],
        'g1': config['model_params']['g1'],
        'g4': config['model_params']['g4']
    }
    
    for k in kwargs:
        pars[k] = kwargs[k]
    
    return pars

