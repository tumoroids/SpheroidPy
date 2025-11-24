import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional

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
    def __init__(self, l=1.0, BA=1.0, sigma=0.9, delta=0.5, beta=0.005, CC=0.1, CD=0.05,
                 m1=1, m2=1, dxi=0.005, dt=0.01, tf=21.0, trecord=None, tol=1e-6,
                 max_newton_iter=50, relaxation=0.8, check_stability=True):
        """
        Initialize the Ward & King model with given parameters.

        Args:
            l (float): Initial spheroid radius (dimensionless). Note: when using __call__,
                       the simulation always starts at l=1.0 and l here is ignored.
            dxi (float): Spatial step size in transformed coordinates
            dt (float): Time step size
            tf (float): Final simulation time
            trecord (list): Time points to record results
            tol (float): Convergence tolerance for Newton-Raphson iterations
            max_newton_iter (int): Maximum number of Newton iterations per time step
            relaxation (float): Under-relaxation factor (0 < relaxation <= 1) for Newton updates
            check_stability (bool): Whether to check for numerical instabilities (NaN, Inf, negative values)
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
        self.max_newton_iter = max_newton_iter
        self.relaxation = max(0.01, min(1.0, relaxation))  # Clamp between 0.01 and 1.0
        self.check_stability = check_stability
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
        self.Lrecord = np.zeros(self.maxsteps+1)  # Outer radius (live radius)
        self.Lrecord[0] = l
        self.Rrecord = np.zeros(self.maxsteps+1)  # Necrotic radius (dead radius)
        self.Rrecord[0] = 0.0  # Initially no necrotic core
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

        # Optional scaling parameters (set via __call__ or set_scaling)
        self.R0_um = None    # characteristic radius in micrometers
        self.R0_m = None     # characteristic radius in meters
        self.time_scale_s = None  # optional physical time scale t0 = R0_m^2 / D in seconds

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
    
    def compute_necrotic_radius(self, C_profile: np.ndarray, l: float) -> float:
        """
        Compute the necrotic radius based on oxygen concentration threshold CD.
        
        The necrotic radius is defined as the radius where oxygen concentration
        drops below the critical death threshold CD. This represents the outer boundary
        of the necrotic core, separating viable tissue (c >= CD) from necrotic tissue (c < CD).
        
        The necrotic radius is the largest radius where c(x) >= CD, i.e., the boundary
        between viable and necrotic regions.
        
        Args:
            C_profile: Oxygen concentration profile (array of length N), from center (index 0) to surface (index N-1)
            l: Current outer radius (spheroid radius)
            
        Returns:
            float: Necrotic radius in physical coordinates. Returns 0.0 if no
                   necrotic core exists (all c >= CD).
        """
        # Convert to physical coordinates: xx = l * xi
        xx = l * self.xi
        
        # Search from outside (surface) to inside (center) to find the boundary
        # The necrotic radius is the largest radius where c >= CD
        # i.e., the first point (from outside) where c < CD
        
        # Find indices where concentration is below CD (necrotic region)
        below_cd_mask = C_profile < self.CD
        
        if not np.any(below_cd_mask):
            # No necrotic core exists if all concentrations are >= CD
            return 0.0
        
        # Find the outermost point with c < CD (starting from surface)
        # This gives us the boundary between viable and necrotic tissue
        below_cd_indices = np.where(below_cd_mask)[0]
        
        if len(below_cd_indices) == 0:
            return 0.0
        
        # Get the outermost (largest index, closest to surface) point with c < CD
        outermost_below_idx = below_cd_indices[-1]
        
        # If the outermost necrotic point is at the surface, the entire spheroid is necrotic
        if outermost_below_idx >= self.N - 1:
            return float(l)
        
        # Interpolate to find the exact radius where c = CD
        # Use linear interpolation between the last point with c >= CD and first with c < CD
        if outermost_below_idx < self.N - 1:
            # Point just outside the necrotic region (c >= CD)
            idx_viable = outermost_below_idx + 1  # First point from outside with c >= CD
            # Point inside the necrotic region (c < CD)
            idx_necrotic = outermost_below_idx      # Last point from outside with c < CD
            
            c_viable = C_profile[idx_viable]
            c_necrotic = C_profile[idx_necrotic]
            x_viable = xx[idx_viable]
            x_necrotic = xx[idx_necrotic]
            
            # Linear interpolation to find x where c = CD
            if abs(c_viable - c_necrotic) > 1e-10:
                alpha = (self.CD - c_necrotic) / (c_viable - c_necrotic)
                necrotic_radius = x_necrotic + alpha * (x_viable - x_necrotic)
            else:
                # If concentrations are very close, use the boundary point
                necrotic_radius = x_viable
            
            return float(necrotic_radius)
        
        return 0.0

    def compute_physical_parameters(self, R0: float = 50e-6, t_unit: float = 86400.0, 
                                     C_infinity: float = 0.2) -> dict:
        """
        Compute physical (dimensional) parameters from dimensionless model parameters.
        
        The Ward & King model is formulated in dimensionless units. This method
        reconstructs the physical parameters (diffusion coefficient, consumption rate, etc.)
        from the dimensionless model parameters.
        
        The scaling relationships are:
        - r̃ = r / R₀ (radius)
        - t̃ = t D / R₀² (time)
        - C̃ = C / C_∞ (oxygen concentration)
        
        where:
        - R₀: characteristic length scale (typically initial spheroid radius)
        - D: oxygen diffusion coefficient [m²/s]
        - C_∞: oxygen concentration at boundary [mol/m³ or mM]
        
        Args:
            R0: Characteristic length scale [m] (default: 50 µm)
            t_unit: Time unit for model [s] (default: 1 day = 86400 s)
            C_infinity: Oxygen concentration at boundary [mol/m³] (default: 0.2 mM ≈ 0.2 mol/m³)
        
        Returns:
            Dictionary with physical parameters:
                - 'D': Oxygen diffusion coefficient [m²/s]
                - 'Q': Oxygen consumption rate [mol/(m³·s)]
                - 'k_growth_max': Maximum cell growth rate [1/s]
                - 'k_death_max': Maximum cell death rate [1/s]
                - 'C_critical_growth': Critical oxygen for growth [mol/m³]
                - 'C_critical_death': Critical oxygen for death [mol/m³]
                - 'R0': Characteristic radius [m]
                - 't_unit': Time unit [s]
        """
        # Calculate diffusion coefficient from scaling
        # D = R₀² / t_unit
        D = R0**2 / t_unit
        
        # Calculate oxygen consumption rate
        # In dimensionless form: β = Q R₀² / (D C_∞)
        # Solving for Q: Q = β D C_∞ / R₀² = β C_∞ / t_unit
        Q = self.beta * C_infinity / t_unit
        
        # Calculate maximum growth rate
        # The mitosis function M(C) gives dimensionless growth rate
        # Physical rate: k_growth = M(C) / t_unit
        # Maximum occurs at C = 1 (normalized concentration)
        M_max = self.mitosis(1.0, self.CC, self.m1)
        k_growth_max = M_max / t_unit
        
        # Calculate maximum death rate
        # Death rate: k_death = BA * death_function(C) / t_unit
        # Maximum occurs at C = 0
        death_max = self.death(0.0, self.BA, self.sigma, self.CD, self.m2)
        k_death_max = death_max / t_unit
        
        # Critical oxygen concentrations (in physical units)
        C_critical_growth = self.CC * C_infinity
        C_critical_death = self.CD * C_infinity
        
        return {
            'D': D,  # m²/s
            'Q': Q,  # mol/(m³·s)
            'k_growth_max': k_growth_max,  # 1/s
            'k_death_max': k_death_max,  # 1/s
            'C_critical_growth': C_critical_growth,  # mol/m³
            'C_critical_death': C_critical_death,  # mol/m³
            'R0': R0,  # m
            't_unit': t_unit,  # s
            'C_infinity': C_infinity  # mol/m³
        }
    
    def get_growth_rate(self, C: float, R0: float = 50e-6, t_unit: float = 86400.0) -> float:
        """
        Calculate physical cell growth rate at given oxygen concentration.
        
        Args:
            C: Oxygen concentration (normalized, 0-1)
            R0: Characteristic length scale [m] (default: 50 µm)
            t_unit: Time unit for model [s] (default: 1 day = 86400 s)
        
        Returns:
            Growth rate [1/s]
        """
        M = self.mitosis(C, self.CC, self.m1)
        return M / t_unit
    
    def get_death_rate(self, C: float, R0: float = 50e-6, t_unit: float = 86400.0) -> float:
        """
        Calculate physical cell death rate at given oxygen concentration.
        
        Args:
            C: Oxygen concentration (normalized, 0-1)
            R0: Characteristic length scale [m] (default: 50 µm)
            t_unit: Time unit for model [s] (default: 1 day = 86400 s)
        
        Returns:
            Death rate [1/s]
        """
        D_rate = self.death(C, self.BA, self.sigma, self.CD, self.m2)
        return D_rate / t_unit
    
    def get_diffusion_coefficient(self, R0: float = 50e-6, t_unit: float = 86400.0) -> float:
        """
        Calculate oxygen diffusion coefficient from scaling.
        
        Args:
            R0: Characteristic length scale [m] (default: 50 µm)
            t_unit: Time unit for model [s] (default: 1 day = 86400 s)
        
        Returns:
            Diffusion coefficient [m²/s]
        """
        return R0**2 / t_unit
    
    def get_consumption_rate(self, R0: float = 50e-6, t_unit: float = 86400.0, 
                            C_infinity: float = 0.2) -> float:
        """
        Calculate oxygen consumption rate per unit volume.
        
        Args:
            R0: Characteristic length scale [m] (default: 50 µm)
            t_unit: Time unit for model [s] (default: 1 day = 86400 s)
            C_infinity: Oxygen concentration at boundary [mol/m³] (default: 0.2 mM)
        
        Returns:
            Consumption rate [mol/(m³·s)]
        """
        D = self.get_diffusion_coefficient(R0, t_unit)
        return self.beta * C_infinity / t_unit
    
    def set_scaling(self, R0_um: float, D_m2_s: Optional[float] = None) -> None:
        """
        Set scaling factors for length (R0) and optionally time (via diffusion coefficient D).
        
        Args:
            R0_um: Characteristic radius in micrometers (µm). Used to scale dimensionless radii.
            D_m2_s: Optional diffusion coefficient [m²/s]. If provided, sets the physical
                    time scale t0 = R0_m^2 / D in seconds for converting model time.
        """
        self.R0_um = float(R0_um)
        self.R0_m = self.R0_um * 1e-6
        if D_m2_s is not None and self.R0_m is not None and self.R0_m > 0:
            self.time_scale_s = (self.R0_m ** 2) / float(D_m2_s)
        else:
            self.time_scale_s = None
    
    def convert_to_physical_units(self, radius_dimless: float, time_dimless: float,
                                  R0: float = 50e-6, t_unit: float = 86400.0) -> tuple:
        """
        Convert dimensionless model units to physical units.
        
        Args:
            radius_dimless: Radius in dimensionless units
            time_dimless: Time in dimensionless units
            R0: Characteristic length scale [m] (default: 50 µm)
            t_unit: Time unit for model [s] (default: 1 day = 86400 s)
        
        Returns:
            Tuple of (radius_physical [m], time_physical [s])
        """
        radius_physical = radius_dimless * R0  # m
        time_physical = time_dimless * t_unit  # s
        return radius_physical, time_physical
    
    def convert_from_physical_units(self, radius_physical: float, time_physical: float,
                                    R0: float = 50e-6, t_unit: float = 86400.0) -> tuple:
        """
        Convert physical units to dimensionless model units.
        
        Args:
            radius_physical: Radius in physical units [m]
            time_physical: Time in physical units [s]
            R0: Characteristic length scale [m] (default: 50 µm)
            t_unit: Time unit for model [s] (default: 1 day = 86400 s)
        
        Returns:
            Tuple of (radius_dimless, time_dimless)
        """
        radius_dimless = radius_physical / R0
        time_dimless = time_physical / t_unit
        return radius_dimless, time_dimless

    def _solve_until_radius(self, target_radius: float) -> float:
        """
        Solve the model until the outer radius reaches the target radius.
        Returns the time taken to reach this radius.
        
        This method runs the simulation without storing full history,
        stopping when the outer radius reaches the target.
        
        Args:
            target_radius: Target outer radius to reach [µm]
            
        Returns:
            Time taken to reach target radius [days]
        """
        import warnings
        
        # Estimate max steps needed (conservative estimate)
        max_steps = int(1000 / self.dt)  # Up to 1000 days
        
        # Main time stepping loop
        for i in range(max_steps):
            self.t += self.dt
            kk = 0
            delnn = np.ones(self.N)
            
            # Newton iterations until convergence
            while np.linalg.norm(delnn, np.inf) > self.tol and kk < self.max_newton_iter:
                kk += 1
                
                # Solve for LIVING CELL DENSITY n (same as solve())
                dldt = self.V[self.N-1]
                km = self.mitosis(self.C[0], self.CC, self.m1)
                kd = self.death(self.C[0], self.BA, self.sigma, self.CD, self.m2)
                
                self.a[0] = 0.0
                self.b[0] = -1.0/self.dt + (km-kd) - 2*self.nn[0]*(km-(1-self.delta)*kd)
                self.c[0] = 0.0
                self.d[0] = (self.nn[0]-self.pnn[0])/self.dt - self.nn[0]*(km-kd) + self.nn[0]**2*(km-(1-self.delta)*kd)
                
                for j in range(1, self.N-1):
                    km = self.mitosis(self.C[j], self.CC, self.m1)
                    kd = self.death(self.C[j], self.BA, self.sigma, self.CD, self.m2)
                    vel = self.xi[j]*dldt/self.l - self.V[j]/self.l
                
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
                
                km = self.mitosis(self.C[self.N-1], self.CC, self.m1)
                kd = self.death(self.C[self.N-1], self.BA, self.sigma, self.CD, self.m2)
                self.a[self.N-1] = 0.0
                self.b[self.N-1] = 1.0
                self.c[self.N-1] = 0.0
                self.d[self.N-1] = -1.0*(self.nn[self.N-1] - (km-kd)*np.exp(self.t*(km-kd))/(
                    (km-kd) - (km-(1-self.delta)*kd)*(1.0-np.exp(self.t*(km-kd)))))
                
                delnn = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.nn += self.relaxation * delnn
                self.nn = np.clip(self.nn, 0.0, 1.0)
                
                # Solve for CELL VELOCITY v
                self.a[0] = 0.0
                self.b[0] = 1.0
                self.c[0] = 0.0
                self.d[0] = -self.V[0]
                
                for j in range(1, self.N-1):
                    km = self.mitosis(self.C[j], self.CC, self.m1)
                    kd = self.death(self.C[j], self.BA, self.sigma, self.CD, self.m2)
                    self.a[j] = -1/(2*self.dxi)
                    self.b[j] = 2.0/self.xi[j]
                    self.c[j] = 1/(2*self.dxi)
                    self.d[j] = -(self.V[j+1]-self.V[j-1])/(2*self.dxi) - 2*self.V[j]/self.xi[j] + self.l*self.nn[j]*(km-(1-self.delta)*kd)
                
                km = self.mitosis(self.C[self.N-1], self.CC, self.m1)
                kd = self.death(self.C[self.N-1], self.BA, self.sigma, self.CD, self.m2)
                self.a[self.N-1] = -1/self.dxi
                self.b[self.N-1] = 2.0 + 1/self.dxi
                self.c[self.N-1] = 0.0
                self.d[self.N-1] = -(self.V[self.N-1]-self.V[self.N-2])/self.dxi - 2*self.V[self.N-1] + self.l*self.nn[self.N-1]*(km-(1-self.delta)*kd)
                
                delV = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.V += self.relaxation * delV
                
                # Solve for OXYGEN CONCENTRATION c
                self.a[0] = 0.0
                self.b[0] = -1.0
                self.c[0] = 1.0
                self.d[0] = -(self.C[1]-self.C[0])
                
                for j in range(1, self.N-1):
                    km = self.mitosis(self.C[j], self.CC, self.m1)
                    kd = self.death(self.C[j], self.BA, self.sigma, self.CD, self.m2)
                    self.a[j] = 1/self.dxi**2 - 1/(self.xi[j]*self.dxi)
                    self.b[j] = -2.0/self.dxi**2 - self.l*self.nn[j]*self.beta*self.m1*self.C[j]**(self.m1-1)*self.CC**self.m1/(self.C[j]**self.m1*self.CC**self.m1)**2
                    self.c[j] = 1/self.dxi**2 + 1/(self.xi[j]*self.dxi)
                    self.d[j] = -(self.C[j+1]-2*self.C[j]+self.C[j-1])/self.dxi**2 - (self.C[j+1]-self.C[j-1])/(self.xi[j]*self.dxi) + self.l**2*self.nn[j]*self.beta*km
                
                self.a[self.N-1] = 0.0
                self.b[self.N-1] = 1.0
                self.c[self.N-1] = 0.0
                self.d[self.N-1] = -(self.C[self.N-1]-1.0)
                
                delO = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.C += self.relaxation * delO
                self.C = np.clip(self.C, 0.0, 1.0)
                
                # Update spheroid radius
                self.l = self.pl + self.dt*self.V[self.N-1]
                
                if self.l <= 0:
                    warnings.warn(f"Negative radius detected at t={self.t:.3f}. Stopping simulation.")
                    break
            
            if self.check_stability:
                if np.any(np.isnan(self.nn)) or np.any(np.isinf(self.nn)) or \
                   np.any(np.isnan(self.C)) or np.any(np.isinf(self.C)) or \
                   np.any(np.isnan(self.V)) or np.any(np.isinf(self.V)) or \
                   not np.isfinite(self.l) or self.l <= 0:
                    break
            
            # Store solution for next time step
            self.pnn = self.nn.copy()
            self.pl = self.l
            
            # Check if we've reached the target radius
            if self.l >= target_radius:
                return self.t
        
        # If we didn't reach the target, return current time
        return self.t

    def __call__(self, t_array: np.ndarray, 
                 # Characteristic length scale (input in micrometers)
                 R0: float = 50.0,
                 # Biological parameters
                 beta: Optional[float] = None, 
                 BA: Optional[float] = None,
                 CD: Optional[float] = None,
                 CC: Optional[float] = None,
                 sigma: Optional[float] = None,
                 delta: Optional[float] = None,
                 m1: Optional[int] = None,
                 m2: Optional[int] = None,
                 # Numerical parameters
                 dt: Optional[float] = None,
                 dxi: Optional[float] = None,
                 tol: Optional[float] = None,
                 max_newton_iter: Optional[int] = None,
                 relaxation: Optional[float] = None,
                 check_stability: Optional[bool] = None,
                 # Plotting
                 plot: bool = False) -> np.ndarray:
        """
        Call the model as a function to get radius evolution.
        
        Similar to GreenspanModel, this method allows calling the model like a function
        to get the live (outer) and necrotic (dead) radii over time.
        
        The simulation always starts with l=1 (dimensionless, corresponding to R0).
        The model works in dimensionless units internally. Radii are returned scaled
        with the provided R0. Here R0 is expected in micrometers (µm) and returned
        radii are in micrometers (µm): R_µm = l_dimless × R0_µm.
        
        Note on time scaling: time in this method is in model units. Mapping to
        physical time requires a diffusion coefficient D via t0 = (R0_m^2 / D).
        
        Args:
            t_array: Array of time points (dimensionless model time)
            R0: Characteristic length scale [µm] (default: 50 µm). This is the reference
                radius used for dimensionless scaling. The model starts at l=1 (1 × R0).
            beta: Optional override for beta parameter (oxygen consumption rate)
            BA: Optional override for BA parameter (death/growth ratio)
            CD: Optional override for CD parameter (critical oxygen for death)
            CC: Optional override for CC parameter (critical oxygen for growth)
            sigma: Optional override for sigma parameter (oxygen sensitivity for death)
            delta: Optional override for delta parameter (volume loss fraction from death)
            m1: Optional override for m1 parameter (Hill coefficient for growth)
            m2: Optional override for m2 parameter (Hill coefficient for death)
            dt: Optional override for time step size
            dxi: Optional override for spatial step size (requires reinitialization of grid)
            tol: Optional override for convergence tolerance
            max_newton_iter: Optional override for maximum Newton iterations
            relaxation: Optional override for under-relaxation factor
            check_stability: Optional override for stability checking flag
            plot: Whether to plot the results (default: False)
            
        Returns:
            Concatenated array of [live_radius_array, necrotic_radius_array] in micrometers [µm]
            Total length is 2 * len(t_array)
        """
        from scipy.interpolate import interp1d
        
        # Interpret R0 in micrometers; store also in meters for physical conversions
        self.R0_um = float(R0)
        self.R0_m = self.R0_um * 1e-6
        
        # Store original parameters
        original_params = {
            'beta': self.beta,
            'BA': self.BA,
            'CD': self.CD,
            'CC': self.CC,
            'sigma': self.sigma,
            'delta': self.delta,
            'm1': self.m1,
            'm2': self.m2,
            'l': self.l,
            'dt': self.dt,
            'dxi': self.dxi,
            'tol': self.tol,
            'max_newton_iter': self.max_newton_iter,
            'relaxation': self.relaxation,
            'check_stability': self.check_stability,
            'tf': self.tf,
            'N': self.N,
            'xi': self.xi.copy() if hasattr(self, 'xi') else None,
            'R0_um': getattr(self, 'R0_um', None),
            'R0_m': getattr(self, 'R0_m', None)
        }
        
        # Update parameters if provided
        if beta is not None:
            self.beta = beta
        if BA is not None:
            self.BA = BA
        if CD is not None:
            self.CD = CD
        if CC is not None:
            self.CC = CC
        if sigma is not None:
            self.sigma = sigma
        if delta is not None:
            self.delta = delta
        if m1 is not None:
            self.m1 = m1
        if m2 is not None:
            self.m2 = m2
        if dt is not None:
            self.dt = dt
        if dxi is not None:
            self.dxi = dxi
            self.N = int(1.0 / self.dxi) + 1
            # Reinitialize grid arrays
            self.xi = np.array([i * self.dxi for i in range(self.N)])
            # Arrays for tridiagonal system solver
            self.a = np.zeros(self.N)
            self.b = np.zeros(self.N)
            self.c = np.zeros(self.N)
            self.d = np.zeros(self.N)
            # Solution arrays
            self.nn = np.ones(self.N)
            self.pnn = np.ones(self.N)
            self.C = np.ones(self.N)
            self.V = np.zeros(self.N)
            # Newton-Raphson update arrays
            self.deln = np.zeros(self.N)
            self.delO = np.zeros(self.N)
            self.delV = np.zeros(self.N)
        if tol is not None:
            self.tol = tol
        if max_newton_iter is not None:
            self.max_newton_iter = max_newton_iter
        if relaxation is not None:
            self.relaxation = max(0.01, min(1.0, relaxation))
        if check_stability is not None:
            self.check_stability = check_stability
        
        # Always start with l=1 (dimensionless, corresponding to R0)
        initial_radius_dimless = 1.0
        
        # Set up for initial simulation (always start at l=1)
        self.l = initial_radius_dimless
        self.pl = initial_radius_dimless
        
        # Set final time to maximum of t_array
        tf = float(np.max(t_array))
        self.tf = tf
        
        # Calculate maxsteps needed
        self.maxsteps = int(self.tf / self.dt)
        
        # Reinitialize arrays for this simulation
        self._reinitialize_arrays()
        
        try:
            # Temporarily disable plotting
            original_trecord = self.trecord
            if not plot:
                self.trecord = []
            
            # Start simulation from l=1 (dimensionless)
            self.t = 0.0
            
            # Initialize arrays for the main simulation (starting from l=1)
            self.Lrecord[0] = self.l
            self.Rrecord[0] = self.compute_necrotic_radius(self.C, self.l)
            self.Crecord[0] = self.C.copy()
            self.Vrecord[0] = self.V.copy()
            self.nrecord[0] = self.nn.copy()
            self.pnn = self.nn.copy()
            self.pl = self.l
            
            # Now run the main simulation
            self.solve()
            
            # Restore original trecord
            self.trecord = original_trecord
            
            # Extract radii at requested time points
            # t_grid is relative to the start of the main simulation
            t_grid = self.dt * np.arange(self.maxsteps + 1)
            
            # Interpolate to get values at requested time points
            # Only use valid data (where Lrecord > 0)
            valid_mask = (self.Lrecord > 0) & np.isfinite(self.Lrecord) & np.isfinite(self.Rrecord)
            if not np.any(valid_mask):
                # Fallback: use nearest neighbors
                idxs = np.array([int(np.argmin(np.abs(t_grid - t))) for t in t_array])
                live_radius = self.Lrecord[idxs]
                necrotic_radius = self.Rrecord[idxs]
            else:
                valid_t = t_grid[valid_mask]
                valid_live = self.Lrecord[valid_mask]
                valid_necrotic = self.Rrecord[valid_mask]
                
                # Create interpolation functions
                if len(valid_t) > 1:
                    interp_live = interp1d(valid_t, valid_live, kind='linear', 
                                         fill_value='extrapolate', bounds_error=False)
                    interp_necrotic = interp1d(valid_t, valid_necrotic, kind='linear',
                                              fill_value=0.0, bounds_error=False)
                    
                    # Interpolate at requested time points
                    live_radius = interp_live(t_array)
                    necrotic_radius = interp_necrotic(t_array)
                else:
                    # Not enough data, use nearest neighbor
                    idxs = np.array([int(np.argmin(np.abs(t_grid - t))) for t in t_array])
                    live_radius = self.Lrecord[idxs]
                    necrotic_radius = self.Rrecord[idxs]
            
            # Ensure non-negative values
            live_radius = np.maximum(live_radius, 0.0)
            necrotic_radius = np.maximum(necrotic_radius, 0.0)
            
            # Convert dimensionless radii to micrometers (multiply by R0 in µm)
            live_radius_um = live_radius * self.R0_um  # [µm]
            necrotic_radius_um = necrotic_radius * self.R0_um  # [µm]
            
            # Concatenate and return (similar to GreenspanModel)
            # Returns values in micrometers [µm]
            result = np.concatenate([live_radius_um, necrotic_radius_um])
            
        finally:
            # Restore original parameters
            self.beta = original_params['beta']
            self.BA = original_params['BA']
            self.CD = original_params['CD']
            self.CC = original_params['CC']
            self.sigma = original_params['sigma']
            self.delta = original_params['delta']
            self.m1 = original_params['m1']
            self.m2 = original_params['m2']
            self.l = original_params['l']
            self.pl = original_params['l']
            self.dt = original_params['dt']
            self.dxi = original_params['dxi']
            self.tol = original_params['tol']
            self.max_newton_iter = original_params['max_newton_iter']
            self.relaxation = original_params['relaxation']
            self.check_stability = original_params['check_stability']
            self.tf = original_params['tf']
            
            # Restore R0 if it existed before
            if original_params['R0_um'] is not None:
                self.R0_um = original_params['R0_um']
            elif hasattr(self, 'R0_um'):
                delattr(self, 'R0_um')
            if original_params['R0_m'] is not None:
                self.R0_m = original_params['R0_m']
            elif hasattr(self, 'R0_m'):
                delattr(self, 'R0_m')
            
            # Restore grid if it was changed
            if original_params['xi'] is not None:
                self.N = original_params['N']
                self.xi = original_params['xi']
                # Reinitialize arrays for tridiagonal solver
                self.a = np.zeros(self.N)
                self.b = np.zeros(self.N)
                self.c = np.zeros(self.N)
                self.d = np.zeros(self.N)
            
            self.maxsteps = int(self.tf / self.dt)
            self._reinitialize_arrays()
        
        return result
    
    def _reinitialize_arrays(self):
        """Reinitialize storage arrays after parameter changes."""
        self.maxsteps = int(self.tf / self.dt)
        self.Lrecord = np.zeros(self.maxsteps + 1)
        self.Lrecord[0] = self.l
        self.Rrecord = np.zeros(self.maxsteps + 1)
        self.Rrecord[0] = 0.0
        self.Crecord = np.zeros((self.maxsteps + 1, self.N))
        self.Crecord[0] = np.ones(self.N)
        self.Vrecord = np.zeros((self.maxsteps + 1, self.N))
        self.Vrecord[0] = np.zeros(self.N)
        self.nrecord = np.zeros((self.maxsteps + 1, self.N))
        self.nrecord[0] = np.ones(self.N)
        self.t = 0.0
        self.pnn = np.ones(self.N)
        self.nn = np.ones(self.N)
        self.C = np.ones(self.N)
        self.V = np.zeros(self.N)

    def solve(self):
        """
        Main solution method that advances the system in time.
        
        Uses an implicit time stepping scheme with Newton-Raphson iterations
        at each time step to handle nonlinearity. Solves sequentially for:
        1. Living cell density (n)
        2. Cell velocity (v) 
        3. Oxygen concentration (c)
        
        Records solution at specified time points and generates plots.
        
        Improvements:
        - Maximum Newton iterations per time step to prevent infinite loops
        - Under-relaxation for better convergence
        - Numerical stability checks (NaN, Inf, negative values)
        - Early termination on numerical instability
        """
        import warnings
        
        # Track convergence issues
        convergence_warnings = []
        stability_issues = []
        
        # Main time stepping loop with progress bar
        pbar = tqdm(range(self.maxsteps), desc="Solving Ward & King model")
        for i in pbar:
            self.t += self.dt  # Advance time
            kk = 0   # Newton iteration counter
            delnn = np.ones(self.N)  # Initialize correction
            
            # Newton iterations until convergence
            while np.linalg.norm(delnn, np.inf) > self.tol and kk < self.max_newton_iter:
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
                
                # Solve and update n with under-relaxation
                delnn = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.nn += self.relaxation * delnn
                
                # Clamp cell density to physical bounds [0, 1]
                self.nn = np.clip(self.nn, 0.0, 1.0)
                
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
                
                # Solve and update v with under-relaxation
                delV = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.V += self.relaxation * delV
                
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
                
                # Solve and update c with under-relaxation
                delO = self.thomas(self.N, self.a, self.b, self.c, self.d)
                self.C += self.relaxation * delO
                
                # Clamp oxygen concentration to physical bounds [0, 1]
                self.C = np.clip(self.C, 0.0, 1.0)
                
                # Update spheroid radius
                self.l = self.pl + self.dt*self.V[self.N-1]
                
                # Ensure radius is positive
                if self.l <= 0:
                    warnings.warn(f"Negative radius detected at t={self.t:.3f}. Stopping simulation.")
                    break
            
            # Check convergence
            if kk >= self.max_newton_iter:
                convergence_warnings.append((i, self.t, kk))
                if len(convergence_warnings) <= 5:  # Limit warning messages
                    warnings.warn(f"Newton iterations did not converge at step {i}, t={self.t:.3f} after {kk} iterations. "
                                f"Residual: {np.linalg.norm(delnn, np.inf):.2e}")
            
            # Numerical stability checks
            if self.check_stability:
                # Check for NaN or Inf
                if np.any(np.isnan(self.nn)) or np.any(np.isinf(self.nn)):
                    stability_issues.append((i, self.t, "cell_density", "NaN/Inf"))
                    warnings.warn(f"NaN/Inf detected in cell density at t={self.t:.3f}. Stopping simulation.")
                    break
                
                if np.any(np.isnan(self.C)) or np.any(np.isinf(self.C)):
                    stability_issues.append((i, self.t, "oxygen_concentration", "NaN/Inf"))
                    warnings.warn(f"NaN/Inf detected in oxygen concentration at t={self.t:.3f}. Stopping simulation.")
                    break
                
                if np.any(np.isnan(self.V)) or np.any(np.isinf(self.V)):
                    stability_issues.append((i, self.t, "velocity", "NaN/Inf"))
                    warnings.warn(f"NaN/Inf detected in velocity at t={self.t:.3f}. Stopping simulation.")
                    break
                
                # Check for unphysical values
                if np.any(self.nn < 0) or np.any(self.nn > 1):
                    # This should be handled by clipping, but log if it happens
                    self.nn = np.clip(self.nn, 0.0, 1.0)
                
                if np.any(self.C < 0) or np.any(self.C > 1):
                    self.C = np.clip(self.C, 0.0, 1.0)
                
                if not np.isfinite(self.l) or self.l <= 0:
                    warnings.warn(f"Invalid radius at t={self.t:.3f}. Stopping simulation.")
                    break
            
            # Update progress information
            pbar.set_description(f"Time: {self.t:.3f}, Iterations: {kk}")
            
            # Compute necrotic radius based on CD threshold
            necrotic_radius = self.compute_necrotic_radius(self.C, self.l)
            
            # Store solution for next time step
            self.pnn = self.nn.copy()
            self.pl = self.l
            self.Lrecord[i+1] = self.l
            self.Rrecord[i+1] = necrotic_radius
            self.Crecord[i+1] = self.C.copy()  # Store concentration
            self.Vrecord[i+1] = self.V.copy()  # Store velocity
            self.nrecord[i+1] = self.nn.copy()  # Store living cell density
            
            # Generate plots at specified time points (only if trecord is not empty)
            if self.trecord and any(abs(np.array(self.trecord) - self.t) < self.dt/10):
                # Scale radius axis if R0 is available
                if getattr(self, 'R0_um', None) is not None:
                    self.xx = (self.l * self.xi) * self.R0_um  # µm
                    x_label = 'Radius [µm]'
                else:
                    self.xx = self.l * self.xi  # model units
                    x_label = 'Radius (model units)'
        
                plt.subplot(2,2,2)
                plt.plot(self.xx, self.nn, 'k', linewidth=2)
                plt.xlabel(x_label)
                plt.ylabel('Living cell density n')
                
                plt.subplot(2,2,3)
                plt.plot(self.xx, self.C, 'k', linewidth=2)
                plt.xlabel(x_label)
                plt.ylabel('Concentration c')
                
                plt.subplot(2,2,4)
                plt.plot(self.xx, self.V, 'k', linewidth=2)
                plt.xlabel(x_label)
                plt.ylabel('Velocity v')
        
        # Plot final radius evolution (only if trecord is not empty)
        if self.trecord:
            plt.subplot(2,2,1)
            t_points_model = self.dt * np.arange(self.maxsteps+1)
            # Time axis scaling if time_scale_s is available
            if getattr(self, 'time_scale_s', None) is not None:
                t_points = (t_points_model * self.time_scale_s) / 86400.0  # days
                t_label = 'Time [days]'
            else:
                t_points = t_points_model
                t_label = 'Time (model units)'
            # Radius scaling if R0 is available
            if getattr(self, 'R0_um', None) is not None:
                L_um = self.Lrecord * self.R0_um
                R_um = self.Rrecord * self.R0_um
                r_label = 'Radius [µm]'
            else:
                L_um = self.Lrecord
                R_um = self.Rrecord
                r_label = 'Radius (model units)'
            plt.plot(t_points, L_um, 'k', linewidth=2, label='Outer radius (live)')
            plt.plot(t_points, R_um, 'r', linewidth=2, linestyle='--', label='Necrotic radius (dead)')
            plt.xlabel(t_label)
            plt.ylabel(r_label)
            plt.legend()
            plt.title('Spheroid Radii Evolution')

            plt.tight_layout()
            plt.show()
        
        # Print summary of convergence and stability issues
        if convergence_warnings:
            print(f"\nConvergence warnings: {len(convergence_warnings)} time steps did not converge within {self.max_newton_iter} iterations")
            if len(convergence_warnings) <= 10:
                print("Affected time steps:", [f"t={t:.3f} (step {step})" for step, t, _ in convergence_warnings])
        
        if stability_issues:
            print(f"\nStability issues: {len(stability_issues)} time steps had numerical instabilities")
            if len(stability_issues) <= 10:
                print(" Issues:", [f"{var} ({issue}) at t={t:.3f}" for _, t, var, issue in stability_issues])
        
        if not convergence_warnings and not stability_issues:
            print("\nSimulation completed successfully without convergence or stability issues.")

if __name__ == "__main__":
    model = WardAndKing()
    model.solve()
