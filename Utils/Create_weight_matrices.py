import numpy as np
import math

def Norm_matrix(N, std_dev, gaussian_height):
    '''Gaussian Normalization pool within each layers'''
    convMat = np.zeros((N, N))

    for row in range(N):
        for col in range(N):
            circular_distance = min(abs(col - row), N - abs(col - row))
            value = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (circular_distance ** 2) / (std_dev ** 2))
            convMat[row, col] = value

        convMat[row] /= np.max(convMat[row])
        convMat[row] = convMat[row] * gaussian_height

    return (convMat + convMat.T)/2.0


def Recurrence_matrix(N, kernel):
    '''Inter-areal recurrent excitation matrix'''
    K = len(kernel)
    convMat = np.zeros((N, N))
    for row in range(N):
        i = 0
        for col in range(int(row - np.floor(K / 2)), int(row + np.ceil(K / 2))):
            convMat[row, np.remainder(col + N, N)] = kernel[i]
            i = i + 1
    return convMat


def RescaleEigenvalues(convMat):
    '''Rescale the eigenvalues recurrent excitation matrix'''
    Utmp, Stmp, Vh = np.linalg.svd(convMat)
    Stmp[Stmp > 1] = 1
    Stmp = Stmp * 1.0
    n_convMat = (Utmp @ np.diag(Stmp) @ Vh)
    return n_convMat

def ReceptiveFields(N, theta, M):
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

def setup_parameters(config, tau=None, kernel=None, N=36, M=None, tauPlus=None, tauY=None, **kwargs):
    '''This function sets up the parameters for the ring model.
    Params:
        config: configuration dictionary from YAML
        tau: baseline membrane time constant for the p/u/s/beta/gamma variables.
             If None, read from config['model_params']['tau'] (falls back to 10e-3 if
             absent). An explicit value overrides the config.
        tauPlus: time constant for the *Plus variables. Resolved like tau via
                 config['model_params']['tauPlus'].
        tauY: membrane time constant for the y1/y4 membrane potentials only. Resolved
              like tau via config['model_params']['tauY']; defaults to `tau` if absent.
        kernel: kernel for the within area recurrent matrix
        N: number of neurons
        M: number of stimuli
        kwargs: additional arguments'''
    # Resolve tau / tauPlus / tauY: explicit argument wins, otherwise read from config.
    if tau is None:
        tau = config['model_params'].get('tau', 10e-3)
    if tauPlus is None:
        tauPlus = config['model_params'].get('tauPlus', 10e-3)
    if tauY is None:
        tauY = config['model_params'].get('tauY', tau)

    if kernel is None:
        kernel = [0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934,
                  0.41472545, -0.073386624, -0.060944743, 0.02807382] # qmf 9 kernel

    # Within area recurrent matrix
    W11 = Recurrence_matrix(N, kernel)
    W11 = (W11.T + W11)/2.0 # making symmetric
    W11 = RescaleEigenvalues(W11) # rescaling eigenvalues to one

    # Normalization matrices (set to small identity)
    # identity_matrix = np.eye(N) * 1e-6
    # Wn1 = identity_matrix
    # Wn2 = identity_matrix
    Wn1 = np.ones((N, N))
    Wn2 = np.ones((N, N))

    W44 = W11 # same within area recurrent matrix
    W14 = W11 @ W11 # connectivity matrix
    W41 = W14.T
    W45 = W14

    if M is None:
        theta = np.arange(0, 2 * np.pi, 2 * np.pi / 360)
        M = len(theta)
    else:
        theta = np.linspace(0, 2 * np.pi, M)

    # Encoding matrix: Receptive fields are raised cosine
    Wzx = ReceptiveFields(N, theta, M)

    pars = {
        'N': N, 'M': M,
        'tauY1': tauY, 'tauY4': tauY,
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
        'tau_x': tau, 'tau_f': config['noise_params']['tau_f'],
        'sigma1': config['model_params']['sigma1'], 'sigma4': config['model_params']['sigma4'],
        'dt': tauPlus/3,
        'W11': W11, 'W44': W44,
        'W14': W14*0.7, 'W41': W41*1.0, 'W45': W45,
        'Wn1': Wn1, 'Wn4': Wn2,
        'Wzx': Wzx*0.7,
        'sigma_f': config['noise_params']['sigma_f'],
        'alpha1': config['model_params']['alpha1'],
        'alpha4': config['model_params']['alpha4'],
        'beta1': config['model_params']['beta1'],
        'beta4': config['model_params']['beta4'],
        'gamma1': config['model_params']['gamma1'],
        'gamma4': config['model_params']['gamma4'],
        'b1': config['model_params']['b1'],
        'b4': config['model_params']['b4'],
        'g1': config['model_params']['g1'],
        'g4': config['model_params']['g4'],
        'Delta_x': config['model_params'].get('Delta_x', 0)
    }

    for k in kwargs:
        pars[k] = kwargs[k]

    return pars
