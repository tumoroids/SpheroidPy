"""
Fit models for necrotic radius analysis.

This module provides various mathematical models for fitting necrotic radius
and outer radius data over time.
"""
import numpy as np


def necrotic_radius_func(t, t_nec, slope, plateau, a):
    """
    Generic function to fit necrotic radius over time.
    
    This function models necrotic radius growth as a combination of linear
    growth and exponential saturation after necrosis starts.
    
    Args:
        t: Time array
        t_nec: Time when necrosis starts
        slope: Linear growth rate after necrosis starts
        plateau: Plateau value for exponential term
        a: Exponential decay rate
        
    Returns:
        Array of necrotic radius values
    """
    t = np.array(t, dtype=float)
    r = np.zeros_like(t, dtype=float)

    mask = t > t_nec
    dt = t[mask] - t_nec

    # linear + exponential term
    linear = slope * dt
    exp_term = plateau * (1 - np.exp(-a * dt))

    r[mask] = linear + exp_term
    return r


def R_out(t, R0, v):
    """
    Outer spheroid radius with linear growth.

    Parameters
    ----------
    t  : array-like
         Time
    R0 : float
         Initial radius
    v  : float
         Radial growth rate

    Returns
    -------
    R : array-like
        Outer radius at time t
    """
    return R0 + v * t


def R_n(R, r_l):
    """
    Necrotic radius for constant oxygen uptake
    (Grimes et al., analytic solution).

    Parameters
    ----------
    R   : array-like
          Outer radius
    r_l : float
          Oxygen diffusion length

    Returns
    -------
    Rn : array-like
         Necrotic radius
    """
    R = np.asarray(R)
    Rn = np.zeros_like(R)

    # Nekrose existiert nur für R >= r_l
    mask = R >= r_l

    # Argument der arccos-Funktion
    arg = 1.0 - 2.0 * r_l**2 / R[mask]**2

    # numerische Sicherheit (Floating Point)
    arg = np.clip(arg, -1.0, 1.0)

    x = (np.arccos(arg) - 2.0 * np.pi) / 3.0

    Rn[mask] = R[mask] * (0.5 - np.cos(x))

    return Rn


def const_uptake_necrotic_radius(t, R0, v, r_l):
    """
    Necrotic radius model with constant oxygen uptake.
    
    This function combines linear outer radius growth (R_out) with
    the constant oxygen uptake model (R_n) to predict necrotic radius.
    
    Args:
        t: Time array
        R0: Initial outer radius (μm)
        v: Radial growth rate (μm/day)
        r_l: Oxygen diffusion length (μm)
        
    Returns:
        Array of necrotic radius values (μm)
    """
    # Calculate outer radius over time
    R_outer = R_out(t, R0, v)
    
    # Calculate necrotic radius from outer radius
    R_necrotic = R_n(R_outer, r_l)
    
    return R_necrotic


def get_fit_model(model_name: str):
    """
    Get fit model function by name.
    
    Args:
        model_name: Name of the model ('generic' or 'const_uptake')
        
    Returns:
        Fit function
        
    Raises:
        ValueError: If model_name is not recognized
    """
    models = {
        'generic': necrotic_radius_func,
        'const_uptake': const_uptake_necrotic_radius
    }
    
    if model_name not in models:
        raise ValueError(
            f"Unknown fit model '{model_name}'. "
            f"Available models: {list(models.keys())}"
        )
    
    return models[model_name]


def get_default_initial_guess(model_name: str, time_array: np.ndarray, 
                              radius_array: np.ndarray, bounds: dict | None = None):
    """
    Get default initial parameter guesses for a fit model.
    
    Args:
        model_name: Name of the model ('generic' or 'const_uptake')
        time_array: Time data array
        radius_array: Radius data array
        bounds: Optional dictionary of parameter bounds
        
    Returns:
        List of initial parameter values
    """
    t_min = np.min(time_array)
    t_max = np.max(time_array)
    r_max = np.max(radius_array)
    
    if model_name == 'generic':
        # Default guesses for generic model
        t_nec_init = t_min + (t_max - t_min) * 0.3  # 30% into time range
        slope_init = r_max / (t_max - t_min) if (t_max - t_min) > 0 else 0.1
        plateau_init = r_max * 0.8
        a_init = 0.1
        
        p0 = [t_nec_init, slope_init, plateau_init, a_init]
        
        # Adjust based on bounds if provided
        if bounds is not None:
            param_names = ['t_nec', 'slope', 'plateau', 'a']
            for i, param_name in enumerate(param_names):
                if param_name in bounds:
                    lower, upper = bounds[param_name]
                    if not (np.isinf(lower) or np.isinf(upper)):
                        # Use midpoint if both bounds are finite
                        p0[i] = (lower + upper) / 2.0
                    elif not np.isinf(lower):
                        p0[i] = lower + 0.1 * abs(lower) if lower != 0 else 0.1
                    elif not np.isinf(upper):
                        p0[i] = upper - 0.1 * abs(upper) if upper != 0 else 0.1
        
        return p0
    
    elif model_name == 'const_uptake':
        # Default guesses for const_uptake model
        R0_init = radius_array[0] if len(radius_array) > 0 else 100.0
        v_init = (r_max - R0_init) / (t_max - t_min) if (t_max - t_min) > 0 else 5.0
        r_l_init = r_max * 0.7  # 70% of max radius
        
        p0 = [R0_init, v_init, r_l_init]
        
        # Adjust based on bounds if provided
        if bounds is not None:
            param_names = ['R0', 'v', 'r_l']
            for i, param_name in enumerate(param_names):
                if param_name in bounds:
                    lower, upper = bounds[param_name]
                    if not (np.isinf(lower) or np.isinf(upper)):
                        # Use midpoint if both bounds are finite
                        p0[i] = (lower + upper) / 2.0
                    elif not np.isinf(lower):
                        p0[i] = lower + 0.1 * abs(lower) if lower != 0 else 0.1
                    elif not np.isinf(upper):
                        p0[i] = upper - 0.1 * abs(upper) if upper != 0 else 0.1
        
        return p0
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_default_bounds(model_name: str, time_array: np.ndarray, 
                       radius_array: np.ndarray):
    """
    Get default parameter bounds for a fit model.
    
    Args:
        model_name: Name of the model ('generic' or 'const_uptake')
        time_array: Time data array
        radius_array: Radius data array
        
    Returns:
        Tuple of (lower_bounds, upper_bounds) lists
    """
    t_min = np.min(time_array)
    t_max = np.max(time_array)
    r_max = np.max(radius_array)
    
    if model_name == 'generic':
        lower = [t_min - (t_max - t_min), -np.inf, 0, 0]
        upper = [t_max, np.inf, r_max * 2, 10]
        return (lower, upper)
    
    elif model_name == 'const_uptake':
        lower = [0, 0, 0]
        upper = [r_max * 2, np.inf, r_max * 2]
        return (lower, upper)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

