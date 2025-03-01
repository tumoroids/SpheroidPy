import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class WardAndKing:
    """
    Implementation of the Ward & King (1997) mathematical model for spheroid growth.
    
    This model simulates avascular tumor growth considering:
    - Living cell density (n)
    - Oxygen concentration (c) 
    - Cell velocity (v)
    - Spheroid radius (l)
    
    The model uses a moving boundary formulation and accounts for:
    - Cell proliferation (mitosis) dependent on oxygen availability
    - Cell death dependent on oxygen availability
    - Oxygen diffusion and consumption
    - Internal pressure-driven cell movement
    """
    def __init__(self, l=50.0, BA=1.0, sigma=0.9, delta=0.5, beta=0.005, CC=0.1, CD=0.05,
                 m1=1, m2=1, dxi=0.001, dt=0.005, tf=21.0, trecord=None, tol=1e-6):
        """
        Initialize the Ward & King model with given parameters.

        Args:
            dxi (float): Spatial step size in transformed coordinates
            dt (float): Time step size
            tf (float): Final simulation time
            trecord (list): Time points to record results
            tol (float): Convergence tolerance for Newton-Raphson iterations
            l (float): Initial spheroid radius
            pl (float): Previous time step radius
            BA (float): Ratio of cell death to growth rates
            sigma (float): Oxygen sensitivity parameter for cell death
            delta (float): Volume loss fraction from cell death
            beta (float): Oxygen consumption rate
            CC (float): Critical oxygen level for cell growth
            CD (float): Critical oxygen level for cell death
            m1 (int): Hill coefficient for growth function
            m2 (int): Hill coefficient for death function
        """
        # Numerical discretization parameters
        self.dxi = dxi    
        self.dt = dt      
        self.t = 0.0      
        self.tf = tf      
        self.trecord = [2,5,10,15,20,30,25, 50, 75, 100] if trecord is None else trecord
        self.tol = tol    
        self.l = l        
        self.pl = l # previous time step radius (initial radius)     

        # Biological model parameters
        self.BA = BA      
        self.sigma = sigma  
        self.delta = delta  
        self.beta = beta  
        self.CC = CC      
        self.CD = CD      
        self.m1 = m1      
        self.m2 = m2      

        # Grid initialization
        self.maxsteps = int(self.tf/self.dt)  
        self.N = int(1.0/self.dxi) + 1   
        
        # Arrays for storing solution history
        self.Lrecord = np.zeros(self.maxsteps+1)  
        self.Lrecord[0] = l
        self.Crecord = np.zeros((self.maxsteps+1, self.N))  # Store concentration history
        self.Crecord[0] = np.ones(self.N)  # Initial concentration is 1 everywhere
        self.Vrecord = np.zeros((self.maxsteps+1, self.N))  # Store velocity history
        self.Vrecord[0] = np.zeros(self.N)  # Initial velocity is 0 everywhere
        self.nrecord = np.zeros((self.maxsteps+1, self.N))  # Store living cell density history
        self.nrecord[0] = np.ones(self.N)  # Initial cell density is 1 everywhere
        
        # Spatial coordinate arrays
        self.xi = np.zeros(self.N)  # Transformed coordinates
        self.xx = np.zeros(self.N)  # Physical coordinates

        # Solution arrays for primary variables
        self.nn = np.ones(self.N)   # Living cell density
        self.pnn = np.ones(self.N)  # Previous time step cell density
        self.C = np.ones(self.N)    # Oxygen concentration
        self.V = np.zeros(self.N)   # Cell velocity

        # Arrays for Newton-Raphson updates
        self.deln = np.zeros(self.N)  
        self.delO = np.zeros(self.N)  
        self.delV = np.zeros(self.N)  

        # Generate transformed coordinate grid
        self.xi = np.array([i * self.dxi for i in range(self.N)])
        
        # Arrays for tridiagonal system solver
        self.a = np.zeros(self.N)  # Lower diagonal
        self.b = np.zeros(self.N)  # Main diagonal
        self.c = np.zeros(self.N)  # Upper diagonal
        self.d = np.zeros(self.N)  # Right hand side
        
        # Initialize plotting
        self.fig = plt.figure(figsize=(10, 8))

    def thomas(self, N, a, b, c, d):
        """
        Thomas Algorithm for solving tridiagonal systems of equations.
        
        Solves the system Ax=d where A is tridiagonal with diagonals (a,b,c).
        Uses an efficient O(N) algorithm avoiding matrix operations.

        Args:
            N (int): System size
            a (ndarray): Lower diagonal elements [a₁, a₂, ..., aₙ₋₁]
            b (ndarray): Main diagonal elements [b₀, b₁, ..., bₙ₋₁]
            c (ndarray): Upper diagonal elements [c₀, c₁, ..., cₙ₋₂]
            d (ndarray): Right hand side vector

        Returns:
            ndarray: Solution vector x
        """
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

    def mitosis(self, c, CC, m1):
        """
        Cell growth rate function dependent on oxygen concentration.
        
        Models Michaelis-Menten type kinetics with Hill coefficient m1.

        Args:
            c (float): Local oxygen concentration
            CC (float): Critical oxygen concentration for growth
            m1 (int): Hill coefficient controlling steepness
            
        Returns:
            float: Growth rate between 0 and 1
        """
        exponent = c**m1
        return exponent / (exponent + CC**m1)

    def death(self, c, BA, sigma, CD, m2):
        """
        Cell death rate function dependent on oxygen concentration.
        
        Models oxygen-dependent cell death with saturation.

        Args:
            c (float): Local oxygen concentration
            BA (float): Maximum death rate
            sigma (float): Oxygen sensitivity parameter
            CD (float): Critical oxygen concentration for death
            m2 (int): Hill coefficient
            
        Returns:
            float: Death rate
        """
        return BA * (1 - sigma*c**m2/(c**m2 + CD**m2))

    def solve(self):
        """
        Main solution method that advances the system in time.
        
        Uses an implicit time stepping scheme with Newton-Raphson iterations
        at each time step to handle nonlinearity. Solves sequentially for:
        1. Living cell density (n)
        2. Cell velocity (v) 
        3. Oxygen concentration (c)
        
        Records solution at specified time points and generates plots.
        """
        # Main time stepping loop with progress bar
        pbar = tqdm(range(self.maxsteps), desc="Solving Ward & King model")
        for i in pbar:
            self.t += self.dt  # Advance time
            kk = 0   # Newton iteration counter
            delnn = np.ones(self.N)  # Initialize correction
            
            # Newton iterations until convergence
            while np.linalg.norm(delnn, np.inf) > self.tol:
                kk += 1
                
                # Solve for LIVING CELL DENSITY n ---------------------------------------------------------------------
                dldt = self.V[self.N-1]
                km = self.mitosis(self.C[0], self.CC, self.m1)
                kd = self.death(self.C[0], self.BA, self.sigma, self.CD, self.m2)
                
                # Set up tridiagonal system for n
                # Left boundary (x=0, symmetry condition)
                self.a[0] = 0.0
                self.b[0] = -1.0/self.dt + (km-kd) - 2*self.nn[0]*(km-(1-self.delta)*kd)
                self.c[0] = 0.0
                self.d[0] = (self.nn[0]-self.pnn[0])/self.dt - self.nn[0]*(km-kd) + self.nn[0]**2*(km-(1-self.delta)*kd)
                
                # Internal nodes (convection-reaction equation)
                for j in range(1, self.N-1):
                    km = self.mitosis(self.C[j], self.CC, self.m1)
                    kd = self.death(self.C[j], self.BA, self.sigma, self.CD, self.m2)
                    vel = self.xi[j]*dldt/self.l - self.V[j]/self.l
                
                    # Upwind differencing based on velocity direction
                    if vel <= 0:
                        self.a[j] = -vel/self.dxi
                        self.b[j] = -1.0/self.dt + vel/self.dxi + (km-kd) - 2*self.nn[j]*(km-(1-self.delta)*kd)
                        self.c[j] = 0.0
                        self.d[j] = (self.nn[j]-self.pnn[j])/self.dt - vel*(self.nn[j] - self.nn[j-1])/self.dxi - self.nn[j]*(km-kd) + self.nn[j]**2*(km-(1-self.delta)*kd)
                    else:
                        self.a[j] = 0.0
                        self.b[j] = -1.0/self.dt - vel/self.dxi + (km-kd) - 2*self.nn[j]*(km-(1-self.delta)*kd)
                        self.c[j] = vel/self.dxi
                        self.d[j] = (self.nn[j]-self.pnn[j])/self.dt - vel*(self.nn[j+1]-self.nn[j])/self.dxi - self.nn[j]*(km-kd) + self.nn[j]**2*(km-(1-self.delta)*kd)
                
                # Right boundary (x=R(t), moving boundary condition)
                km = self.mitosis(self.C[self.N-1], self.CC, self.m1)
                kd = self.death(self.C[self.N-1], self.BA, self.sigma, self.CD, self.m2)
                self.a[self.N-1] = 0.0
                self.b[self.N-1] = 1.0
                self.c[self.N-1] = 0.0
                self.d[self.N-1] = -1.0*(self.nn[self.N-1] - (km-kd)*np.exp(self.t*(km-kd))/(
                    (km-kd) - (km-(1-self.delta)*kd)*(1.0-np.exp(self.t*(km-kd)))))
                
                # Solve and update n
                delnn = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.nn += delnn
                
                # Solve for CELL VELOCITY v ---------------------------------------------------------------------
                # Left boundary (v=0 at x=0 by symmetry)
                self.a[0] = 0.0
                self.b[0] = 1.0
                self.c[0] = 0.0
                self.d[0] = -self.V[0]
                
                # Internal nodes (mass conservation equation)
                for j in range(1, self.N-1):
                    km = self.mitosis(self.C[j], self.CC, self.m1)
                    kd = self.death(self.C[j], self.BA, self.sigma, self.CD, self.m2)
                    self.a[j] = -1/(2*self.dxi)
                    self.b[j] = 2.0/self.xi[j]
                    self.c[j] = 1/(2*self.dxi)
                    self.d[j] = -(self.V[j+1]-self.V[j-1])/(2*self.dxi) - 2*self.V[j]/self.xi[j] + self.l*self.nn[j]*(km-(1-self.delta)*kd)
                
                # Right boundary (stress-free condition)
                km = self.mitosis(self.C[self.N-1], self.CC, self.m1)
                kd = self.death(self.C[self.N-1], self.BA, self.sigma, self.CD, self.m2)
                self.a[self.N-1] = -1/self.dxi
                self.b[self.N-1] = 2.0 + 1/self.dxi
                self.c[self.N-1] = 0.0
                self.d[self.N-1] = -(self.V[self.N-1]-self.V[self.N-2])/self.dxi - 2*self.V[self.N-1] + self.l*self.nn[self.N-1]*(km-(1-self.delta)*kd)
                
                # Solve and update v
                delV = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.V += delV
                
                # Solve for OXYGEN CONCENTRATION c ---------------------------------------------------------------------
                # Left boundary (symmetry condition)
                self.a[0] = 0.0
                self.b[0] = -1.0
                self.c[0] = 1.0
                self.d[0] = -(self.C[1]-self.C[0])
                
                # Internal nodes (reaction-diffusion equation)
                for j in range(1, self.N-1):
                    km = self.mitosis(self.C[j], self.CC, self.m1)
                    kd = self.death(self.C[j], self.BA, self.sigma, self.CD, self.m2)
                    self.a[j] = 1/self.dxi**2 - 1/(self.xi[j]*self.dxi)
                    self.b[j] = -2.0/self.dxi**2 - self.l*self.nn[j]*self.beta*self.m1*self.C[j]**(self.m1-1)*self.CC**self.m1/(self.C[j]**self.m1*self.CC**self.m1)**2
                    self.c[j] = 1/self.dxi**2 + 1/(self.xi[j]*self.dxi)
                    self.d[j] = -(self.C[j+1]-2*self.C[j]+self.C[j-1])/self.dxi**2 - (self.C[j+1]-self.C[j-1])/(self.xi[j]*self.dxi) + self.l**2*self.nn[j]*self.beta*km
                
                # Right boundary (c=1 at x=R(t))
                self.a[self.N-1] = 0.0
                self.b[self.N-1] = 1.0
                self.c[self.N-1] = 0.0
                self.d[self.N-1] = -(self.C[self.N-1]-1.0)
                
                # Solve and update c
                delO = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.C += delO
                
                # Update spheroid radius
                self.l = self.pl + self.dt*self.V[self.N-1]
            
            # Update progress information
            pbar.set_description(f"Time: {self.t:.3f}, Iterations: {kk}")
            
            # Store solution for next time step
            self.pnn = self.nn.copy()
            self.pl = self.l
            self.Lrecord[i+1] = self.l
            self.Crecord[i+1] = self.C.copy()  # Store concentration
            self.Vrecord[i+1] = self.V.copy()  # Store velocity
            self.nrecord[i+1] = self.nn.copy()  # Store living cell density
            
            # Generate plots at specified time points
            if any(abs(np.array(self.trecord) - self.t) < self.dt/10):
                self.xx = self.l * self.xi
        
                plt.subplot(2,2,2)
                plt.plot(self.xx, self.nn, 'k', linewidth=2)
                plt.xlabel('Radius x')
                plt.ylabel('Living cell density n')
                
                plt.subplot(2,2,3)
                plt.plot(self.xx, self.C, 'k', linewidth=2)
                plt.xlabel('Radius x')
                plt.ylabel('Concentration c')
                
                plt.subplot(2,2,4)
                plt.plot(self.xx, self.V, 'k', linewidth=2)
                plt.xlabel('Radius x')
                plt.ylabel('Velocity v')
        
        # Plot final radius evolution
        plt.subplot(2,2,1)
        t_points = self.dt * np.arange(self.maxsteps+1)
        plt.plot(t_points, self.Lrecord, 'k', linewidth=2)
        plt.xlabel('Time t')
        plt.ylabel('Spheroid radius l(t)')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model = WardAndKing()
    model.solve()
