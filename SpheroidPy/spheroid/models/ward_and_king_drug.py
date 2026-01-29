import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class DrugSpheroidModel:
    """
    Implementation of the Ward & King (2003) mathematical model for spheroid growth with drug effects.
    
    This model extends the basic Ward & King model to include:
    - Living cell density (n)
    - Oxygen concentration (c) 
    - Cell velocity (v)
    - Spheroid radius (S)
    - Drug concentration (w)
    
    The model accounts for:
    - Cell proliferation and death dependent on both oxygen and drug concentration
    - Oxygen diffusion and consumption
    - Drug diffusion and metabolism
    - Internal pressure-driven cell movement
    """
    def __init__(self, S0=142.0, BA=1.0, sigma=0.9, delta=0.5, beta=0.005, 
                 cc=0.1, cd=0.05, m1=1, m2=1, K=50, wc=2.0, alpha=1.0,
                 dxi=0.001, dt=0.005, tf=50.0, trecord=None, tol=1e-6,
                 kinetics_type="linear"):
        """
        Initialize the Drug-modified spheroid model with given parameters.

        Args:
            S0 (float): Initial spheroid radius
            BA (float): Ratio of cell death to growth rates (B/A)
            sigma (float): Oxygen sensitivity parameter for cell death
            delta (float): Volume loss fraction from cell death
            beta (float): Oxygen consumption rate
            cc (float): Critical oxygen level for cell growth
            cd (float): Critical oxygen level for cell death
            m1, m2 (int): Hill coefficients
            K (float): Drug effectiveness parameter
            wc (float): Critical drug concentration (for Michaelis-Menten)
            alpha (float): Drug diffusion parameter
            kinetics_type (str): Either "linear" or "michaelis_menten"
        """
        # Numerical discretization parameters
        self.dxi = dxi    
        self.dt = dt      
        self.t = 0.0      
        self.tf = tf
        self.trecord = [2,5,10,15,20,30] if trecord is None else trecord
        self.tol = tol    
        self.S = S0
        self.pS = S0  # previous time step radius

        # Biological model parameters
        self.BA = BA      
        self.sigma = sigma  
        self.delta = delta  
        self.beta = beta  
        self.cc = cc      
        self.cd = cd      
        self.m1 = m1      
        self.m2 = m2
        self.K = K
        self.wc = wc
        self.alpha = alpha
        self.kinetics_type = kinetics_type

        # Grid initialization
        self.maxsteps = int(self.tf/self.dt)
        self.N = int(1.0/self.dxi) + 1
        
        # Arrays for storing solution history
        self.Srecord = np.zeros(self.maxsteps+1)
        self.Srecord[0] = S0
        self.nrecord = np.zeros((self.maxsteps+1, self.N))
        self.crecord = np.zeros((self.maxsteps+1, self.N))
        self.wrecord = np.zeros((self.maxsteps+1, self.N))
        self.vrecord = np.zeros((self.maxsteps+1, self.N))
        
        # Initialize current solution arrays
        self.xi = np.linspace(0, 1, self.N)
        self.n = np.ones(self.N)
        self.pn = np.ones(self.N)
        self.c = np.ones(self.N)
        self.w = np.zeros(self.N)
        self.v = np.zeros(self.N)
        
        # Arrays for Newton-Raphson updates
        self.deln = np.zeros(self.N)
        self.delc = np.zeros(self.N)
        self.delw = np.zeros(self.N)
        self.delv = np.zeros(self.N)
        
        # Arrays for tridiagonal solver
        self.a = np.zeros(self.N)
        self.b = np.zeros(self.N)
        self.c = np.zeros(self.N)
        self.d = np.zeros(self.N)

    def f_drug(self, w):
        """Drug effect function f(w) - either linear or Michaelis-Menten"""
        if self.kinetics_type == "linear":
            return w
        else:  # Michaelis-Menten
            return w / (w + self.wc)

    def km(self, c):
        """Cell growth rate function"""
        return c**self.m1 / (c**self.m1 + self.cc**self.m1)

    def kd(self, c, w):
        """Cell death rate function including drug effects"""
        natural_death = self.BA * (1 - self.sigma * c**self.m2 / (c**self.m2 + self.cd**self.m2))
        drug_death = self.K * self.f_drug(w)
        return natural_death + drug_death

    def thomas(self, N, a, b, c, d):
        """Thomas Algorithm for solving tridiagonal systems"""
        x = np.zeros(N)
        bb = b.copy()
        dd = d.copy()
        
        # Forward elimination
        for i in range(1, N):
            ff = a[i] / bb[i-1]
            bb[i] = bb[i] - c[i-1]*ff
            dd[i] = dd[i] - dd[i-1]*ff
            
        # Back substitution
        x[N-1] = dd[N-1]/bb[N-1]
        for i in range(N-2, -1, -1):
            x[i] = (dd[i] - c[i]*x[i+1]) / bb[i]
        return x

    def solve(self):
        """
        Main solution method that advances the system in time using the non-dimensionalized equations.
        Uses an implicit time stepping scheme with Newton-Raphson iterations.
        """
        pbar = tqdm(range(self.maxsteps), desc="Solving Drug-modified spheroid model")
        
        for i in pbar:
            self.t += self.dt
            kk = 0
            deln = np.ones(self.N)
            
            # Newton iterations until convergence
            while np.linalg.norm(deln, np.inf) > self.tol:
                kk += 1
                
                # Solve for living cell density n
                dSdt = self.v[self.N-1]
                
                # Left boundary (x=0, symmetry)
                self.a[0] = 0.0
                self.b[0] = -1.0
                self.c[0] = 1.0
                self.d[0] = -(self.n[1] - self.n[0])
                
                # Internal nodes
                for j in range(1, self.N-1):
                    km = self.km(self.c[j])
                    kd = self.kd(self.c[j], self.w[j])
                    vel = self.xi[j]*dSdt/self.S - self.v[j]/self.S
                    
                    if vel <= 0:
                        self.a[j] = -vel/self.dxi
                        self.b[j] = -1.0/self.dt + vel/self.dxi + (km-kd)
                        self.c[j] = 0.0
                        self.d[j] = (self.n[j]-self.pn[j])/self.dt - self.n[j]*(km-kd)
                    else:
                        self.a[j] = 0.0
                        self.b[j] = -1.0/self.dt - vel/self.dxi + (km-kd)
                        self.c[j] = vel/self.dxi
                        self.d[j] = (self.n[j]-self.pn[j])/self.dt - self.n[j]*(km-kd)
                
                # Right boundary (x=S(t))
                self.a[self.N-1] = 0.0
                self.b[self.N-1] = 1.0
                self.c[self.N-1] = 0.0
                self.d[self.N-1] = 1.0 - self.n[self.N-1]
                
                # Solve and update n
                deln = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.n += deln
                
                # Solve for drug concentration w
                # Left boundary (symmetry)
                self.a[0] = 0.0
                self.b[0] = -1.0
                self.c[0] = 1.0
                self.d[0] = -(self.w[1] - self.w[0])
                
                # Internal nodes
                for j in range(1, self.N-1):
                    self.a[j] = self.alpha/(self.dxi**2) - self.alpha/(self.xi[j]*self.dxi)
                    self.b[j] = -2.0*self.alpha/(self.dxi**2) - self.K*self.n[j]
                    self.c[j] = self.alpha/(self.dxi**2) + self.alpha/(self.xi[j]*self.dxi)
                    self.d[j] = -self.K*self.n[j]*self.f_drug(self.w[j])
                
                # Right boundary (w=w0(t))
                self.a[self.N-1] = 0.0
                self.b[self.N-1] = 1.0
                self.c[self.N-1] = 0.0
                self.d[self.N-1] = -(self.w[self.N-1] - 1.0)  # Assuming w0(t)=1
                
                # Solve and update w
                delw = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.w += delw
                
                # Update spheroid radius
                self.S = self.pS + self.dt*self.v[self.N-1]
            
            # Store solution for next time step
            self.pn = self.n.copy()
            self.pS = self.S
            
            # Record solutions
            self.Srecord[i+1] = self.S
            self.nrecord[i+1] = self.n.copy()
            self.wrecord[i+1] = self.w.copy()
            self.vrecord[i+1] = self.v.copy()
            
            # Update progress information
            pbar.set_description(f"Time: {self.t:.3f}, Iterations: {kk}")
            
            # Generate plots at specified time points
            if any(abs(np.array(self.trecord) - self.t) < self.dt/10):
                self.plot_current_state()

    def plot_current_state(self):
        """Plot the current state of the system"""
        plt.figure(figsize=(12, 10))
        
        # Plot radius evolution
        plt.subplot(2,2,1)
        t_points = self.dt * np.arange(len(self.Srecord))
        plt.plot(t_points, self.Srecord, 'k', linewidth=2)
        plt.xlabel('Time t')
        plt.ylabel('Spheroid radius S(t)')
        
        # Plot cell density
        plt.subplot(2,2,2)
        x = self.S * self.xi
        plt.plot(x, self.n, 'k', linewidth=2)
        plt.xlabel('Radius x')
        plt.ylabel('Living cell density n')
        
        # Plot drug concentration
        plt.subplot(2,2,3)
        plt.plot(x, self.w, 'k', linewidth=2)
        plt.xlabel('Radius x')
        plt.ylabel('Drug concentration w')
        
        # Plot velocity
        plt.subplot(2,2,4)
        plt.plot(x, self.v, 'k', linewidth=2)
        plt.xlabel('Radius x')
        plt.ylabel('Velocity v')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_alpha_values(alpha_values, tf=2.0):
        """
        Compare spheroid growth for different alpha values.
        
        Args:
            alpha_values (list): List of alpha values to compare
            tf (float): Final time for simulation
        """
        plt.figure(figsize=(10, 8))
        
        # Run simulation for each alpha value
        for alpha in alpha_values:
            model = DrugSpheroidModel(
                S0=142.0,  # Initial radius
                BA=1.0,    # B/A ratio
                sigma=0.9, # Oxygen sensitivity
                delta=0.5, # Volume loss fraction
                beta=0.005,# Oxygen consumption
                cc=0.1,    # Critical oxygen for growth
                cd=0.05,   # Critical oxygen for death
                m1=1,      # Hill coefficient
                m2=1,      # Hill coefficient
                K=50,      # Drug effectiveness (for Mitomycin C)
                alpha=alpha,# Drug diffusion parameter
                tf=tf,     # Final time
                kinetics_type="linear"
            )
            
            # Solve without showing intermediate plots
            with tqdm(range(model.maxsteps), desc=f"α={alpha}") as pbar:
                for i in pbar:
                    model.t += model.dt
                    kk = 0
                    deln = np.ones(model.N)
                    
                    while np.linalg.norm(deln, np.inf) > model.tol:
                        kk += 1
                        
                        # Solve for living cell density n
                        dSdt = model.v[model.N-1]
                        
                        # Left boundary (x=0, symmetry)
                        model.a[0] = 0.0
                        model.b[0] = -1.0
                        model.c[0] = 1.0
                        model.d[0] = -(model.n[1] - model.n[0])
                        
                        # Internal nodes
                        for j in range(1, model.N-1):
                            km = model.km(model.c[j])
                            kd = model.kd(model.c[j], model.w[j])
                            vel = model.xi[j]*dSdt/model.S - model.v[j]/model.S
                            
                            if vel <= 0:
                                model.a[j] = -vel/model.dxi
                                model.b[j] = -1.0/model.dt + vel/model.dxi + (km-kd)
                                model.c[j] = 0.0
                                model.d[j] = (model.n[j]-model.pn[j])/model.dt - model.n[j]*(km-kd)
                            else:
                                model.a[j] = 0.0
                                model.b[j] = -1.0/model.dt - vel/model.dxi + (km-kd)
                                model.c[j] = vel/model.dxi
                                model.d[j] = (model.n[j]-model.pn[j])/model.dt - model.n[j]*(km-kd)
                        
                        # Right boundary (x=S(t))
                        model.a[model.N-1] = 0.0
                        model.b[model.N-1] = 1.0
                        model.c[model.N-1] = 0.0
                        model.d[model.N-1] = 1.0 - model.n[model.N-1]
                        
                        # Solve and update n
                        deln = model.thomas(model.N, model.a, model.b, model.c, model.d)
                        model.n += deln
                        
                        # Solve for drug concentration w
                        # Left boundary (symmetry)
                        model.a[0] = 0.0
                        model.b[0] = -1.0
                        model.c[0] = 1.0
                        model.d[0] = -(model.w[1] - model.w[0])
                        
                        # Internal nodes
                        for j in range(1, model.N-1):
                            model.a[j] = model.alpha/(model.dxi**2) - model.alpha/(model.xi[j]*model.dxi)
                            model.b[j] = -2.0*model.alpha/(model.dxi**2) - model.K*model.n[j]
                            model.c[j] = model.alpha/(model.dxi**2) + model.alpha/(model.xi[j]*model.dxi)
                            model.d[j] = -model.K*model.n[j]*model.f_drug(model.w[j])
                        
                        # Right boundary (w=w0(t))
                        model.a[model.N-1] = 0.0
                        model.b[model.N-1] = 1.0
                        model.c[model.N-1] = 0.0
                        model.d[model.N-1] = -(model.w[model.N-1] - 1.0)  # Assuming w0(t)=1
                        
                        # Solve and update w
                        delw = model.thomas(model.N, model.a, model.b, model.c, model.d)
                        model.w += delw
                        
                        # Update spheroid radius
                        model.S = model.pS + model.dt*model.v[model.N-1]
                        
                        # Update progress
                        pbar.set_description(f"α={alpha}, t={model.t:.3f}, iter={kk}")
                    
                    # Store solutions
                    model.pn = model.n.copy()
                    model.pS = model.S
                    model.Srecord[i+1] = model.S
            
            # Plot results
            t_points = model.dt * np.arange(len(model.Srecord))
            if alpha == 0.1:
                plt.plot(t_points, model.Srecord, 'k-', linewidth=2, label=f'α = {alpha}')
            else:
                plt.plot(t_points, model.Srecord, 'k--', linewidth=1, label=f'α = {alpha}')
        
        plt.xlabel('time (t)')
        plt.ylabel('spheroid radius (S)')
        plt.ylim(125, 165)
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Reproduce the plot from the paper
    alpha_values = [0.1, 1, 10, 100, 1000, 10000]
    DrugSpheroidModel.compare_alpha_values(alpha_values, tf=2.0) 