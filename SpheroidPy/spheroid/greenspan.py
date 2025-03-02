# Parameters in our "reduced" parameterization
Q = .7
R1 = 200.0
gamma = 3
s = 0.4
R0 = 80.0

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# Define colors as constants
COLORS = {
    'proliferation': "#00A600",
    'inhibited': "#B9FC00",
    'necrotic': "#EB2427"
}

@dataclass
class ModelParameters:
    """Parameters for the Greenspan spheroid growth model."""
    Q: float      # Balance between nutrient and inhibitor concentration (< 1)
    gamma: float  # Balance between cell growth and necrosis loss
    s: float      # Cell proliferation rate [1/d]
    R1: float     # Radius at which necrotic core emerges [µm]
    R0: float     # Initial spheroid size [µm]
    delta: float = 2.0  # Treatment effect parameter

class GreenspanModel:
    """
    A mathematical model for spheroid growth based on Greenspan's work.
    
    This model simulates the growth of spheroids considering three distinct regions:
    - Proliferating region (outer)
    - Inhibited region (middle)
    - Necrotic region (core)
    
    The model accounts for nutrient diffusion, waste accumulation, and cell state transitions.
    
    The class can be called like a function:
    >>> model = GreenspanModel()
    >>> t_array = np.linspace(0, 20, 100)
    >>> radii = model(t_array, Q=0.7, gamma=3, s=0.4, R1=200, R0=80)
    This returns a concatenated array [Ro_array, Ri_array, Rn_array]
    """
    
    def __init__(self, Q: float = 0.7, gamma: float = 3, s: float = 0.4, R1: float = 200.0, R0: float = 80.0):
        """
        Initialize the Greenspan model with given parameters.
        
        Args:
            Q: Balance between nutrient and inhibitor concentration (< 1)
            gamma: Balance between cell growth and necrosis loss
            s: Cell proliferation rate [1/d]
            R1: Radius at which necrotic core emerges [µm]
            R0: Initial spheroid size [µm]
        """
        self.params = ModelParameters(Q=Q, gamma=gamma, s=s, R1=R1, R0=R0)
        
        # Initialize derived parameters
        self.derived_params = {
            'omega1': 20.0,   # Minimum nutrient concentration
            'omega2': 100.0,  # Maximum nutrient concentration
            'A': 12.0,       # Nutrient consumption rate
            'k': 1000.0,     # Nutrient diffusion coefficient
            'beta1': 300.0,  # Critical waste concentration
            'kappa': 500.0   # Waste diffusion coefficient
        }
        
        # Calculate P parameter
        self.derived_params['P'] = (
            self.derived_params['A'] 
            / (self.derived_params['k'] * (self.derived_params['omega2'] - self.derived_params['omega1'])) 
            * self.derived_params['beta1'] 
            * self.derived_params['kappa'] 
            / (self.params.Q ** 2)
        )
        
        # Drug treatment parameters
        self.drug_params = {
            'D_sigma': 1.0,    # Drug diffusion coefficient
            'sigma_0': 0.2,    # Initial drug concentration
            'sigma_min': 0.05, # Minimum effective drug concentration
            'lambda_sigma': 1e-2  # Drug decay rate
        }

    def __call__(self, t_array: np.ndarray, Q: Optional[float] = None, gamma: Optional[float] = None, 
                 s: Optional[float] = None, R1: Optional[float] = None, R0: Optional[float] = None) -> np.ndarray:
        """
        Call the model as a function to get radius evolution.
        
        Args:
            t_array: Array of time points [days]
            Q: Optional override for Q parameter
            gamma: Optional override for gamma parameter
            s: Optional override for s parameter
            R1: Optional override for R1 parameter
            R0: Optional override for R0 parameter
            
        Returns:
            Concatenated array of [Ro_array, Ri_array, Rn_array]
        """
        # Store original parameters
        original_params = self.params
        
        # Update parameters if provided
        if any(param is not None for param in [Q, gamma, s, R1, R0]):
            self.params = ModelParameters(
                Q=Q if Q is not None else original_params.Q,
                gamma=gamma if gamma is not None else original_params.gamma,
                s=s if s is not None else original_params.s,
                R1=R1 if R1 is not None else original_params.R1,
                R0=R0 if R0 is not None else original_params.R0
            )
        
        try:
            # Simulate and concatenate results
            Ro, Ri, Rn = self.simulate(t_array)
            result = np.concatenate([Ro, Ri, Rn])
        finally:
            # Restore original parameters
            self.params = original_params
        
        return result

    def calc_phi_eta(self, R: float) -> Tuple[float, float]:
        """
        Calculate the dimensionless radii Phi and Eta.
        
        Args:
            R: Current radius [µm]
            
        Returns:
            Tuple of (Phi, Eta) representing dimensionless radii
        """
        try:
            R = float(R[0])
        except (TypeError, IndexError):
            R = float(R)

        # Phase 1: No inhibited or necrotic regions
        if R <= min(self.params.Q, 1.0) * self.params.R1:
            return 0.0, 0.0

        # Phase 2: Inhibited region present, no necrotic core
        elif R <= self.params.R1:
            Phi = np.sqrt(1 - (self.params.Q * self.params.R1) ** 2 / R ** 2)
            return Phi, 0.0

        # Phase 3: Both inhibited and necrotic regions present
        else:
            # Calculate Eta (necrotic radius ratio)
            coefs = [2 * R ** 2, -3 * R ** 2, 0.0, R ** 2 - self.params.R1 ** 2]
            roots = np.roots(coefs)
            valid_roots = [root for root in roots if np.isreal(root) and 0.0 <= root <= 1.0]
            
            if not valid_roots:
                raise ValueError("No valid roots found for eta")
            Eta = float(np.real(valid_roots[0]))

            # Calculate Phi (inhibited radius ratio)
            coefs = [
                R ** 2,
                0.0,
                (self.params.Q ** 2) * (self.params.R1 ** 2) - (R ** 2) * (1 + 2 * (Eta ** 3)),
                2 * Eta ** 3 * R ** 2
            ]
            roots = np.roots(coefs)
            valid_roots = [root for root in roots if np.isreal(root) and Eta <= root <= 1.0]
            
            if valid_roots:
                Phi = float(np.real(valid_roots[0]))
            else:
                Phi = np.sqrt(1 - (self.params.Q * self.params.R1) ** 2 / R ** 2)
                
            return Phi, Eta

    def greenspan_ode(self, t: float, R: float) -> float:
        """
        Calculate the growth rate dR/dt at time t.
        
        Args:
            t: Current time [days]
            R: Current radius [µm]
            
        Returns:
            Growth rate dR/dt [µm/day]
        """
        phi, eta = self.calc_phi_eta(R)
        return self.params.s / 3 * R * (1 - phi**3 - 3 * self.params.gamma * eta**3)

    def simulate(self, t_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate spheroid growth over a time array.
        
        Args:
            t_array: Array of time points [days]
            
        Returns:
            Tuple of (outer_radius, inhibited_radius, necrotic_radius) arrays
        """
        solution = solve_ivp(
            self.greenspan_ode,
            [t_array[0], t_array[-1]],
            [self.params.R0],
            dense_output=True,
            method='RK45',
            rtol=1e-6
        )
        
        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")
            
        def get_all_radii(t):
            R = solution.sol(t)[0]
            phi, eta = self.calc_phi_eta(R)
            return R, R * phi, R * eta
            
        return np.vectorize(get_all_radii)(t_array)

    def plot_evolution(self, t_array: np.ndarray, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plot and optionally save the evolution of spheroid radii over time.
        
        Args:
            t_array: Array of time points [days]
            save_path: Optional path to save the plot
            
        Returns:
            Tuple of (outer_radius, inhibited_radius, necrotic_radius) arrays
        """
        Ro_array, Ri_array, Rn_array = self.simulate(t_array)
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_array, Ro_array, color=COLORS['proliferation'], label='Outer (Proliferating)')
        plt.plot(t_array, Ri_array, color=COLORS['inhibited'], label='Inhibited')
        plt.plot(t_array, Rn_array, color=COLORS['necrotic'], label='Necrotic')
        plt.xlabel('Time (days)')
        plt.ylabel('Radius (µm)')
        plt.title('Spheroid Growth Evolution')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return Ro_array, Ri_array, Rn_array

    def create_growth_animation(self, t_array: np.ndarray, save_path: str = "spheroid_growth.mp4"):
        """
        Create and save an animation of the spheroid growth.
        
        Args:
            t_array: Array of time points [days]
            save_path: Path to save the animation
        """
        Ro_array, Ri_array, Rn_array = self.simulate(t_array)
        r_max = np.max(Ro_array) * 1.25
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(frame):
            ax.clear()
            ax.set_xlim(-r_max, r_max)
            ax.set_ylim(-r_max, r_max)
            
            # Plot regions
            if Rn_array[frame] > 0:
                ax.fill(*circle(Rn_array[frame]), color=COLORS['necrotic'], alpha=0.7, label='Necrotic')
            if Ri_array[frame] > 0:
                ax.fill(*circle(Ri_array[frame]), color=COLORS['inhibited'], alpha=0.7, label='Inhibited')
            ax.fill(*circle(Ro_array[frame]), color=COLORS['proliferation'], alpha=0.7, label='Proliferating')
            
            ax.set_title(f'Time: {t_array[frame]:.1f} days')
            ax.legend(loc='upper right')
            ax.grid(True)
            ax.set_aspect('equal')
            
        anim = animation.FuncAnimation(fig, update, frames=len(t_array), interval=50)
        anim.save(save_path, writer='ffmpeg', fps=10)
        plt.close()

    def plot_until(self, t_end: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plot the evolution of all radii up to a specified end time.
        
        Args:
            t_end: End time in days
            num_points: Number of time points to use (default: 100)
            
        Returns:
            Tuple of (outer_radius, inhibited_radius, necrotic_radius) arrays
        """
        t_array = np.linspace(0, t_end, num_points)
        Ro_array, Ri_array, Rn_array = self.simulate(t_array)
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_array, Ro_array, color=COLORS['proliferation'], label='Outer (Proliferating)', linewidth=2)
        plt.plot(t_array, Ri_array, color=COLORS['inhibited'], label='Inhibited', linewidth=2)
        plt.plot(t_array, Rn_array, color=COLORS['necrotic'], label='Necrotic', linewidth=2)
        plt.xlabel('Time (days)')
        plt.ylabel('Radius (µm)')
        plt.title('Spheroid Growth Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return Ro_array, Ri_array, Rn_array

def circle(r: float) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to generate circle coordinates."""
    theta = np.linspace(0, 2 * np.pi, 100)
    return r * np.sin(theta), r * np.cos(theta)

if __name__ == '__main__':

    # Create time array
    t_array = np.linspace(0, 20, 100)

    # Method 1: Create with default parameters and call
    model = GreenspanModel()
    radii = model(t_array)  # Uses default parameters

    # Method 2: Create with specific parameters
    model = GreenspanModel(Q=0.7, gamma=3, s=0.4, R1=200, R0=80)
    radii = model(t_array)  # Uses parameters from initialization

    # Method 3: Override parameters during call
    model = GreenspanModel()
    radii = model(t_array, Q=0.8, gamma=4)  # Overrides only Q and gamma

    # The returned array 'radii' contains:
    # [Ro_array, Ri_array, Rn_array] concatenated
    # Length is 3 * len(t_array)

    # To get individual radius arrays back:
    n = len(t_array)
    Ro = radii[:n]
    Ri = radii[n:2*n]
    Rn = radii[2*n:]

    print(Ro)
    print(Ri)
    print(Rn)

    model.plot_until(20)
    plt.savefig('spheroid_growth.png')
