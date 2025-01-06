
## The model
from scipy.integrate import solve_ivp
# from Utils.weight_matrix import *
import autograd.numpy as np
from autograd import jacobian
import copy
import tqdm


def relu(x,rectify):
        return np.maximum(rectify,x) 

class RingModel:
    '''
    This class defines the model of the ring network.
    with functions to calculate the Jacobian of the system.
    '''
    def __init__(self, params, target_angle=180, simulate_firing_rates=False,rectify=1e-6,epsilon=1e-6):
        self._params = params
        self._simulate_firing_rates = simulate_firing_rates
        self.epsilon = epsilon # to avoid division by zero, does not have an affect on the coherence curve and communication subspace
        self.rectify = rectify # this changes the shape of the coherence curve but does not affect the communication subspace
        self.M = params['M']
        self.N = params['N']
        self.target_angle = target_angle
        num_neurons_per_area = 4
        num_area =2
        if self.simulate_firing_rates:
            self.num_var = num_area * 2 * num_neurons_per_area + 2* num_area  # 20 variables 
            self.jacobian_dimension = num_area * 2 * num_neurons_per_area  # We have to remove the input gain, feedback gain neurons and their corresponding firing rates from the Jacobian
            # Because, if we include them, the Jacobian will have eigenvalues at 0 or > 0.
        else:
            self.num_var = num_area * num_neurons_per_area + 2*num_area  #  12 variables
            self.jacobian_dimension = num_area * num_neurons_per_area
        self.J = None
        self.contrast = None
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value
    
    @property
    def simulate_firing_rates(self):
        return self._simulate_firing_rates
    
    @simulate_firing_rates.setter
    def simulate_firing_rates(self, value):
        self._simulate_firing_rates = value
        
    
        
    def single_area_computation(self,y,yPlus,u,uPlus,p,pPlus,s,sPlus,beta,gamma,z,yPlus_next,n,n_next):
        '''
        Function to compute the dynamics of a single area.  
        Inputs:
        y,u,p,s : array-like
            The membrane potentials of the excitatory neurons.
        yPlus,uPlus,pPlus,sPlus : array-like
            The firing rates of the excitatory neurons.
        beta,gamma : array-like
            The feedback gain and input gain.
        z : array-like
            The input drive to the area.
        yPlus_next : array-like
            The firing rates of the excitatory neurons in the next area.
        n : int
            The area number.
        n_next : int
            The next area number.
        Returns:
        output : array-like
            The time derivatives of the membrane potentials, firing rates, feedback gain and input gain. 
        '''
        # Extract parameters based on area number
        tau_y = self.params[f'tauY{n}']
        tau_p = self.params[f'tauP{n}']
        tau_s = self.params[f'tauS{n}']
        tau_u = self.params[f'tauU{n}']
        tau_beta = self.params[f'tauBeta{n}']
        tau_gamma = self.params[f'tauGamma{n}']

        Wnn = self.params[f'W{n}{n}']
        Wnn_next = self.params[f'W{n}{n_next}']
        alpha = self.params[f'alpha{n}']
        sigma = self.params[f'sigma{n}']
        Wn = self.params[f'Wn{n}']  # Changed from N{n} to Wn{n}
        beta_0 = self.params[f'beta{n}']
        gamma_0 = self.params[f'gamma{n}']
        b = self.params[f'b{n}']
        g = self.params[f'g{n}']

        # Compute dynamics
        dy = (1 / tau_y) * (-y + (beta*b) * z + (1 / (1 + pPlus)) * (np.matmul(Wnn, np.sqrt(yPlus)) + (gamma*g) * np.matmul(Wnn_next, np.sqrt(yPlus_next))))
        
        du = (1 / tau_u) * (-u + (sigma*b)**2 + np.matmul(Wn, yPlus * uPlus**2))
        
        dp = (1 / tau_p) * (-p + (g*np.matmul(Wnn_next, np.sqrt(yPlus_next))) / ( sPlus ) + uPlus + p * uPlus + alpha *  tau_u * du)
        
        ds = (1 / tau_s) * (-s + np.sqrt(yPlus+self.epsilon))
        
        dbeta = (1 / tau_beta) * (-beta + beta_0)
        dgamma = (1 / tau_gamma) * (-gamma + gamma_0)

        return dy, du, dp, ds, dbeta, dgamma
        
    
    
    # @staticmethod
    def dynm_func(self, t, Y, x):
        '''
        Function to compute the dynamics of the system.
        Inputs:
        t : float
            The time variable.
        Y : array-like
            The state vector of the system.
        x : array-like
            The input to the system.
        Returns:
        dY : array-like
            The time derivatives of the state vector.   
        '''
        N = self.N
        Y = np.squeeze(Y)
        y5Plus = np.zeros((N))
        # unpack variables
        if self.simulate_firing_rates: # 20 variables
            y1,y1Plus, y4,y4Plus, u1,u1Plus, u4,u4Plus, p1,p1Plus, p4,p4Plus,s1,s1Plus,s4,s4Plus,beta1,beta4, gamma1,gamma4 = [Y[i*N:(i+1)*N] for i in range(self.num_var)]
        else: # 12 variables
            y1,y4, u1,u4, p1,p4,s1,s4, beta1,beta4,gamma1,gamma4 = [Y[i*N:(i+1)*N] for i in range(self.num_var)]
        
        if self.simulate_firing_rates:
            dy1Plus = (1 / self.params['tauYPlus1']) * (-y1Plus + relu(y1, self.rectify) ** 2)
            dy4Plus = (1 / self.params['tauYPlus4']) * (-y4Plus + relu(y4, self.rectify) ** 2)
            du1Plus = (1 / self.params['tauUPlus1']) * (-u1Plus + np.sqrt(relu(u1, self.rectify)))
            du4Plus = (1 / self.params['tauUPlus4']) * (-u4Plus + np.sqrt(relu(u4, self.rectify)))
            dp1Plus = (1 / self.params['tauPPlus1']) * (-p1Plus + relu(p1, self.rectify))
            dp4Plus = (1 / self.params['tauPPlus4']) * (-p4Plus + relu(p4, self.rectify))
            ds1Plus = (1 / self.params['tauSPlus1']) * (-s1Plus + relu(s1, self.rectify))
            ds4Plus = (1 / self.params['tauSPlus4']) * (-s4Plus + relu(s4, self.rectify))
        else:
            y1Plus = relu(y1, self.rectify) ** 2 
            y4Plus = relu(y4, self.rectify) ** 2 
            u1Plus = np.sqrt(relu(u1, self.rectify))  
            u4Plus = np.sqrt(relu(u4, self.rectify))
            p1Plus = relu(p1, self.rectify)
            p4Plus = relu(p4, self.rectify)
            s1Plus = relu(s1, self.rectify)
            s4Plus = relu(s4, self.rectify)
            

        
        z1 = np.matmul(x, self.params['Wzx'].T) # input to V1
        z4 = np.matmul( self.params['W41'] , y1Plus) # input to V4
        
        dy1,du1,dp1,ds1,dbeta1,dgamma1 = self.single_area_computation(y1,y1Plus,u1,u1Plus,p1,p1Plus,s1,s1Plus,beta1,gamma1,z1,y4Plus,n=1,n_next=4) # V1
        dy4,du4,dp4,ds4,dbeta4,dgamma4 = self.single_area_computation(y4,y4Plus,u4,u4Plus,p4,p4Plus,s4,s4Plus,beta4,gamma4,z4,y5Plus,n=4,n_next=5) # V4
           

        if self.simulate_firing_rates:
            dY = np.concatenate([dy1, dy1Plus, dy4, dy4Plus, du1, du1Plus, du4, du4Plus, dp1, dp1Plus, dp4, dp4Plus, ds1, ds1Plus, ds4, ds4Plus, dbeta1, dbeta4, dgamma1, dgamma4])
        else:
            dY = np.concatenate([dy1, dy4, du1, du4, dp1, dp4,ds1,ds4, dbeta1, dbeta4, dgamma1, dgamma4])
        return dY  
    
    def wrapper_func_autograd(self,Y):
        """
        Wrapper function for autograd to calculate Jacobian.
        
        Parameters:
        Y : array-like
            The state vector of the system.
        
        Returns:
        array-like
            The time derivatives of the state vector.
        """
        t = 0.0
        x = np.zeros((self.M))
        x[self.target_angle] = self.contrast
        return self.dynm_func(t, Y, x)
    
    def get_Jacobian(self,  c,initial_conditions, method='RK45', t_span=[0, 5]):
        '''
        Function to calculate the Jacobian of the system.
        Inputs:
        params : dictionary
            Dictionary containing the parameters of the system.
        initial_conditions : array-like
            The initial conditions of the system.
        c : float
            The contrast value.
        method : string
            The method to use for the integration.
        t_span : array-like
            The time span for the integration.
        Returns:
        J : array-like
            The Jacobian of the system.
        ss : array-like
            The steady state of the system.
        
        
        '''
        # Get steady states
        N = self.N
        ss = get_steady_states(self, c, initial_conditions, t_span, method, Jacobian=True)
        ss = np.array(ss).flatten()  # Flatten the steady state array
        # add a small constant to the ss of y1Plus and y4Plus
        # ss[1*N:2*N] = ss[1*N:2*N] + 1e-7
        # ss[3*N:4*N] = ss[3*N:4*N] + 1e-7 # TODO: change this
        # ss = ss + (1e-7)
        # Set contrast for Jacobian calculation
        self.contrast = c

        # Calculate Jacobian
        jac_func = jacobian(self.wrapper_func_autograd)
        J = jac_func(ss)

        start_range1 = self.jacobian_dimension*self.N
        end_range1 = self.num_var*self.N
        indices = list(range(start_range1,end_range1))
        J = np.delete(J, indices, axis=0)
        J = np.delete(J, indices, axis=1)
        self.J = J
        return J,ss
    
    def get_dynamics(self, params, initial_conditions, c, method='RK45', t_span=[0, 5], stimulus_onset=False, onset_time=2.5, initial_contrast=0.1):
        '''
        Function to calculate the dynamics of the system.
        Inputs:
        params : dictionary
            Dictionary containing the parameters of the system.
        initial_conditions : array-like
            The initial conditions of the system.
        c : float
            The contrast value.
        method : string
            The method to use for the integration.
        t_span : array-like
            The time span for the integration.
        stimulus_onset : bool
            Whether the stimulus onset is to be considered.
        onset_time : float
            The team of the sitemulius onset.
        initial_contrast : float
            The initial contrast value.
        Returns:
        soln: array-like
            The solution of the system.
        '''
        N = self.N
        M = self.M
        self.contrast = c

        x = np.zeros((M))
        x[self.target_angle] = c
        t_eval = np.arange(t_span[0], t_span[1], params['dt'])
        
        if stimulus_onset:
            # Define a function to handle the contrast change
            def contrast_change(t, y):
                return initial_contrast if t < onset_time else c

            # Modify the dynm_func to include the contrast change
            def modified_dynm_func(t, y):
                current_contrast = contrast_change(t, y)
                x[self.target_angle] = current_contrast
                return self.dynm_func(t, y, x)

            # Use the modified function for solve_ivp
            sol = solve_ivp(modified_dynm_func, t_span, initial_conditions, method=method, t_eval=t_eval,
                            vectorized=True, rtol=1e-10, atol=1e-10)
        else:
            # Use the original function if stimulus_onset is False
            sol = solve_ivp(self.dynm_func, t_span, initial_conditions, method=method, t_eval=t_eval,
                            args=(x,), vectorized=True, rtol=1e-10, atol=1e-10)

        soln = sol.y.reshape(self.num_var, N, -1)

        return [soln[i, int(N/2), :] for i in range(self.num_var)]
    
    def analytical_solution(self, c):
        x = np.zeros((self.params['M']))
        x[self.target_angle] = c
        z1 = np.matmul(x, self.params['Wzx'].T)
        norm = np.matmul(self.params['Wn1'], z1**2)
        y1 = np.sqrt((z1**2 / (self.params['sigma1']**2 + norm)))
        y1Plus = relu(y1, self.rectify) **2
        z4 = np.matmul(self.params['W41'], y1Plus)
        norm2 = np.matmul(self.params['Wn4'], z4**2)
        y4 = np.sqrt(z4**2 / (self.params['sigma4']**2 + norm2))
        y4Plus = relu(y4, self.rectify) ** 2 
        return y1Plus, y4Plus
    

    
    
    
# Utils Functions
def get_steady_states(model,c, initial_conditions, t_span=[0, 5], method='RK45', Jacobian=True):
    ''' This function computes the steady state of the model given the input contrast c and initial conditions.
    Inputs:
    model: RingModel object
    params: dictionary of parameters
    c: input contrast
    initial_conditions: initial conditions of the model
    t_span: time span for the integration
    method: integration method
    Jacobian: Boolean
    Output:
    steady_state: steady state of the model 
    '''
    params = model.params
    N = params['N']
    M = params['M']
    x = np.zeros((M))
    x[model.target_angle] = c
    t_eval = np.arange(t_span[0], t_span[1], params['dt'])
    
    
    sol = solve_ivp(model.dynm_func, t_span, initial_conditions, method=method, t_eval=t_eval, 
                    args=(x,), vectorized=True, rtol=1e-10, atol=1e-10)
    
    soln = sol.y.reshape(model.num_var, N, -1)
    tol = 1e-6
    steady_state_reached = np.all(np.abs(soln[:,:,-1] - soln[:,:,-2]) < tol)
    
    if not steady_state_reached:
        raise ValueError('Solution did not converge to a steady state within the given time span')
    
    steady_state = [soln[i,:,-1] for i in range(model.num_var)]
    
    if Jacobian:
        return steady_state
    else:
        return [soln[i,int(N/2),-1] for i in range(model.num_var)]


def Contrast_response_curve(model,params,c_vals,gamma_vals,method, t_span,initial_conditions,g,fb_gain,input_gain_beta1,input_gain_beta4):
    '''
    Function to compute the contrast response curve of the model.
    Inputs:
    model: RingModel object
    params: dictionary of parameters
    c_vals: array-like
        The contrast values.
    gamma_vals: array-like
        The gamma1/beta1/beta4 values.
    method: string
        The method to use for the integration.
    t_span: array-like
        The time span for the integration.
    initial_conditions: array-like
        The initial conditions of the system.
    fb_gain: bool
        Whether to vary the feedback gain.
    input_gain_beta1: bool
        Whether to vary the input gain beta1.
    input_gain_beta4: bool
        Whether to vary the input gain beta4.
    Returns:
    steady_states: dictionary
        The steady states of the system
        '''

    steady_states = {}
    for gamma in tqdm(gamma_vals):
        updated_model = copy.deepcopy(model)
        if fb_gain:
            updated_model.params['gamma1'] = gamma
        elif input_gain_beta1:
            updated_model.params['beta1'] = gamma
        elif input_gain_beta4:
            updated_model.params['beta4'] = gamma

        # Create a copy of the model with updated parameters
        # updated_model = copy.deepcopy(model)
        # updated_model.params = updated_params 
        updated_model.params['g1'] = g
        for c in c_vals:
            steady_state = get_steady_states(updated_model,c,initial_conditions, t_span,method, Jacobian=False)
            steady_states[(gamma, c)] = steady_state

    return steady_states





