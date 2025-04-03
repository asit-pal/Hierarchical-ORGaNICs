# Standard library imports
# Third-party imports
from scipy.integrate import solve_ivp
import autograd.numpy as np
from autograd import jacobian
from tqdm import tqdm

def relu(x,rectify):
        return np.maximum(rectify,x) 

class RingModel:
    def __init__(self, params, target_angle=180, simulate_firing_rates=False, rectify=1e-6, epsilon=1e-6):
        self._params = params
        self._simulate_firing_rates = simulate_firing_rates
        self.epsilon = epsilon
        self.rectify = rectify
        self.M = params['M']
        self.N = params['N']
        self.target_angle = target_angle
        
        num_neurons_per_area = 4
        num_area = 3  # V1, V4, V5
        if self.simulate_firing_rates:
            self.num_var = num_area * 2 * num_neurons_per_area   # 30 variables
            self.jacobian_dimension = num_area * 2 * num_neurons_per_area
        else:
            self.num_var = num_area * num_neurons_per_area   # 18 variables
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
    
    def single_area_computation(self,y,yPlus,u,uPlus,p,pPlus,s,sPlus,z,yPlus1,yPlus2,n,n1,n2):
        tau_y = self.params[f'tauY{n}']
        tau_p = self.params[f'tauP{n}']
        tau_s = self.params[f'tauS{n}']
        tau_u = self.params[f'tauU{n}']
        
        
        Wnn = self.params[f'W{n}{n}']
        Wnn1 = self.params[f'W{n}{n1}']
        Wnn2 = self.params[f'W{n}{n2}']
        alpha = self.params[f'alpha{n}']
        sigma = self.params[f'sigma{n}']
        Wn = self.params[f'Wn{n}']
        beta = self.params[f'beta{n}']
        b = self.params[f'b{n}']
        g1 = self.params[f'g{n}{n1}']
        g2 = self.params[f'g{n}{n2}']
        gamma1 = self.params[f'gamma{n}{n1}']
        gamma2 = self.params[f'gamma{n}{n2}']
        
         # Compute dynamics
        dy = (1 / tau_y) * (-y + (beta*b) * z + (1 / (1 + pPlus)) * (np.matmul(Wnn, np.sqrt(yPlus)) + (gamma1*g1) * np.matmul(Wnn1, np.sqrt(yPlus1)) +  (gamma2*g2) * np.matmul(Wnn2, np.sqrt(yPlus2))))
        
        du = (1 / tau_u) * (-u + (sigma*b)**2 + np.matmul(Wn, yPlus * uPlus**2))
        
        dp = (1 / tau_p) * (-p + (g1*np.matmul(Wnn1, np.sqrt(yPlus1))) / ( sPlus ) + (g2*np.matmul(Wnn2, np.sqrt(yPlus2))) / ( sPlus ) + uPlus + p * uPlus + alpha *  tau_u * du)
        
        ds = (1 / tau_s) * (-s + np.sqrt(yPlus+self.epsilon))
        
        # dbeta = (1 / tau_beta) * (-beta + beta_0)
        # dgamma1 = (1 / tau_gamma) * (-gamma1 + gamma1_0)
        # dgamma2 = (1 / tau_gamma) * (-gamma2 + gamma2_0)    

        return dy, du, dp, ds

    def dynm_func(self, t, Y, x):
        N = self.N
        Y = np.squeeze(Y)
        
        # Unpack variables for three areas (V1, V4, V5)
        if self.simulate_firing_rates:  # 30 variables
            y1,y1Plus, y4,y4Plus, y5,y5Plus, u1,u1Plus, u4,u4Plus, u5,u5Plus, p1,p1Plus, p4,p4Plus, p5,p5Plus, s1,s1Plus, s4,s4Plus, s5,s5Plus = [Y[i*N:(i+1)*N] for i in range(self.num_var)]
        else:  # 18 variables
            y1,y4,y5, u1,u4,u5, p1,p4,p5, s1,s4,s5 = [Y[i*N:(i+1)*N] for i in range(self.num_var)]
        
        # Calculate firing rates if not simulating them
        if self.simulate_firing_rates:
            dy1Plus = (1 / self.params['tauYPlus1']) * (-y1Plus + relu(y1, self.rectify) ** 2)
            dy4Plus = (1 / self.params['tauYPlus4']) * (-y4Plus + relu(y4, self.rectify) ** 2)
            dy5Plus = (1 / self.params['tauYPlus5']) * (-y5Plus + relu(y5, self.rectify) ** 2)
            du1Plus = (1 / self.params['tauUPlus1']) * (-u1Plus + np.sqrt(relu(u1, self.rectify)))
            du4Plus = (1 / self.params['tauUPlus4']) * (-u4Plus + np.sqrt(relu(u4, self.rectify)))
            du5Plus = (1 / self.params['tauUPlus5']) * (-u5Plus + np.sqrt(relu(u5, self.rectify)))
            dp1Plus = (1 / self.params['tauPPlus1']) * (-p1Plus + relu(p1, self.rectify))
            dp4Plus = (1 / self.params['tauPPlus4']) * (-p4Plus + relu(p4, self.rectify))
            dp5Plus = (1 / self.params['tauPPlus5']) * (-p5Plus + relu(p5, self.rectify))
            ds1Plus = (1 / self.params['tauSPlus1']) * (-s1Plus + relu(s1, self.rectify))
            ds4Plus = (1 / self.params['tauSPlus4']) * (-s4Plus + relu(s4, self.rectify))
            ds5Plus = (1 / self.params['tauSPlus5']) * (-s5Plus + relu(s5, self.rectify))
        else:
            y1Plus = relu(y1, self.rectify) ** 2 
            y4Plus = relu(y4, self.rectify) ** 2
            y5Plus = relu(y5, self.rectify) ** 2
            u1Plus = np.sqrt(relu(u1, self.rectify))
            u4Plus = np.sqrt(relu(u4, self.rectify))
            u5Plus = np.sqrt(relu(u5, self.rectify))
            p1Plus = relu(p1, self.rectify)
            p4Plus = relu(p4, self.rectify)
            p5Plus = relu(p5, self.rectify)
            s1Plus = relu(s1, self.rectify)
            s4Plus = relu(s4, self.rectify)
            s5Plus = relu(s5, self.rectify)
        
        

        # Input drives
        z1 = np.matmul(x, self.params['Wzx'].T)  # input to V1
        z4 = np.matmul(self.params['W41'], y1Plus)  # input to V4 from V1
        z5 = np.matmul(self.params['W51'], y1Plus)  # input to V5 from V1 (using same weights as V4)

        # Compute dynamics for each area
        # Note: For V4 and V5, we use the same parameters as they're functionally identical
        # Computing dynamics for V1
        dy1,du1,dp1,ds1 = self.single_area_computation(
            y1,y1Plus,u1,u1Plus,p1,p1Plus,s1,s1Plus,z1,
            y4Plus, y5Plus,  # Combined feedback from both V4 and V5
            n=1,n1=4,n2=5)  # Using V4 parameters for both feedbacks
        
        # Computing dynamics for V4
        dy4,du4,dp4,ds4 = self.single_area_computation(
            y4,y4Plus,u4,u4Plus,p4,p4Plus,s4,s4Plus,z4,
            y1Plus, y5Plus,  # No feedback to V4
            n=4,n1=1,n2=5)
        
        # Computing dynamics for V5
        dy5,du5,dp5,ds5 = self.single_area_computation(
            y5,y5Plus,u5,u5Plus,p5,p5Plus,s5,s5Plus,z5,
            y1Plus, y4Plus,  # No feedback to V5
            n=5,n1=1,n2=4)  # Using V4 parameters for V5

        if self.simulate_firing_rates:
            dY = np.concatenate([
                dy1,dy1Plus, dy4,dy4Plus, dy5,dy5Plus,
                du1,du1Plus, du4,du4Plus, du5,du5Plus,
                dp1,dp1Plus, dp4,dp4Plus, dp5,dp5Plus,
                ds1,ds1Plus, ds4,ds4Plus, ds5,ds5Plus,
            
            ])
        else:
            dY = np.concatenate([
                dy1,dy4,dy5, du1,du4,du5, dp1,dp4,dp5,
                ds1,ds4,ds5
            ])
        return dY
    
    def wrapper_func_autograd(self, Y):
        """
        Wrapper function for autograd to calculate Jacobian.
        """
        t = 0.0
        x = np.zeros((self.M))
        x[self.target_angle] = self.contrast
        return self.dynm_func(t, Y, x)

    def get_steady_states(self, contrast, initial_conditions, method, t_span=[0, 5], Jacobian=True):
        """
        Calculate steady states of the system.
        
        Args:
            contrast: Input contrast value
            initial_conditions: Initial state of the system
            method: Integration method for solve_ivp
            t_span: Time span for integration [start, end]
            Jacobian: If True, returns full steady state array, if False returns center neuron values
        
        Returns:
            steady_state: Array of steady state values
        """
        params = self.params
        N = params['N']
        M = params['M']
        x = np.zeros((M))
        x[self.target_angle] = contrast
        t_eval = np.arange(t_span[0], t_span[1], params['dt'])
        
        
        # Solve system
        sol = solve_ivp(self.dynm_func, t_span, initial_conditions, 
                        method=method, t_eval=t_eval, args=(x,),
                        vectorized=True, rtol=1e-10, atol=1e-10)
        
        # Reshape solution
        soln = sol.y.reshape(self.num_var, N, -1)
        
        # Check convergence
        tol = 1e-6
        steady_state_reached = np.all(np.abs(soln[:,:,-1] - soln[:,:,-2]) < tol)
        
        if not steady_state_reached:
            raise ValueError('Solution did not converge to a steady state within the given time span')
        
        if Jacobian:
            # Return full steady state for Jacobian calculation
            steady_state = [soln[i,:,-1] for i in range(self.num_var)]
            return steady_state
        else:
            # Return center neuron values only
            return [soln[i,int(N/2),-1] for i in range(self.num_var)]

    def get_Jacobian(self, contrast, initial_conditions, method, t_span=[0, 5]):
        """
        Calculate the Jacobian matrix at steady state.
        
        Args:
            contrast: Input contrast value
            initial_conditions: Initial state of the system
            method: Integration method for solve_ivp
            t_span: Time span for integration [start, end]
        
        Returns:
            J: Jacobian matrix
            ss: Steady state values
        """
        # Store contrast for wrapper_func_autograd
        self.contrast = contrast
        
        # Get steady states
        ss = self.get_steady_states(contrast, initial_conditions, method, t_span, Jacobian=True)
        ss = np.array(ss).flatten()
        
        # Calculate Jacobian using autograd
        jac_func = jacobian(self.wrapper_func_autograd)
        J = jac_func(ss)
        
        self.J = J
        return J, ss
    
    def get_Jacobian_augmented(self, contrast, initial_conditions, method, t_span=[0, 5]):
        """
        Compute the augmented Jacobian that includes additional filtered noise variables for the membrane potentials.

        In the original system (when simulate_firing_rates is True) the state vector x is:
            x = [y1, y1Plus, y4, y4Plus, y5, y5Plus, u1, u1Plus, u4, u4Plus, u5, u5Plus, 
                 p1, p1Plus, p4, p4Plus, p5, p5Plus, s1, s1Plus, s4, s4Plus, s5, s5Plus]
        with total length n = (jacobian_dimension * N). Here, the membrane potentials (without the 'Plus'
        suffix) appear at indices 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22.

        We wish to augment the system by adding an additional noise variable f for each membrane potential.
        The augmented dynamics are:
            x_dot = J x + (f / tau_x)
            f_dot = -f / tau_f + (noise)
        so that the (deterministic) Jacobian for the augmented system is a block matrix:
                    [ J        I/tau_x ]
            J_aug = [ 0      -I/tau_f  ]
        where I/tau_x is inserted only in the rows corresponding to the membrane potentials.

        The new augmented state is [x; f] and the steady state for f is assumed to be zero.

        Args:
            contrast: Contrast value.
            initial_conditions: Initial condition for the original state vector x.
            method: Integration method (default 'RK45').
            t_span: Time span for integration.

        Returns:
            J_aug: Augmented Jacobian matrix.
            ss_aug: Augmented steady state vector ([ss; 0]).
        """
        J, ss = self.get_Jacobian(contrast, initial_conditions, method, t_span)
        N = self.N
        n = self.jacobian_dimension * N  # original state dimension
        
        # Define the indices of the membrane potentials in the original state vector
        if self.simulate_firing_rates:
            membrane_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        else:
            membrane_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        
        # Define the indices of the noise variables in the augmented state vector
        m = len(membrane_indices) * N
        
        # Construct the upper-right block A: shape(n,m)
        # For each membrane potential, we add an identity block divided by tau_x
        A = np.zeros((n, m))
        for idx, mem in enumerate(membrane_indices):
            row_start = mem * N
            row_end = (mem + 1) * N
            col_start = idx * N
            col_end = (idx + 1) * N
            A[row_start:row_end, col_start:col_end] = np.eye(N) / self.params['tau_x']
        
        # Lower-left block (B) is zeros, shape(m,n)
        B = np.zeros((m, n))
        
        # Lower right block C is -I/tau_f shape (m,m)
        C = -np.eye(m) / self.params['tau_f']
        
        # Construct the augmented Jacobian by forming the block matrix
        #        [J  A]
        # J_aug = [B  C]
        J_aug = np.block([[J, A],
                          [B, C]])
        
        # The augmented steady state is the original steady state with zeros appended for the new f variables
        ss_aug = np.concatenate([ss, np.zeros(m)])
        
        return J_aug, ss_aug
    