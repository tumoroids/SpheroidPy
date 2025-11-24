"""
Fluorescence intensity profile analysis utilities.

This module provides comprehensive analysis functions for fluorescence intensity
profiles, including statistical measures, inflection point detection, threshold
analysis, and functional zone identification.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple
from .geometry import find_inflection_point

import numpy as np
from scipy.optimize import curve_fit

def sigmoid_increasing(r, r0, s, C):
    return C + 1.0 / (1.0 + np.exp(-(r - r0) / s))

def sigmoid_decreasing(r, r0, s, C):
    return C + 1.0 / (1.0 + np.exp((r - r0) / s))



def fit_sigmoid_auto(
    r,
    I,
    I_std=None,
    uncertainty_thr=0.3,
    plateau_thr=0.5,
    s_min=None,
    s_max=.4,
    n_outer_frac=0.05,
    increasing=None
    ):
    """
    Fit an increasing or decreasing sigmoid automatically (A=1).
    - Normalize only by max(I)
    - Weighted fit using I_std
    - Fit accepted only if std at r0 < uncertainty_thr (optional)
    - Optional valid range for s: s_min <= s <= s_max
    - C bounds are automatically set to the outermost points ± std
    - Optional plateau_thr criterion
    
    Args:
        increasing: If True, force increasing sigmoid; if False, force decreasing; if None, auto-detect
    """
    I_max = np.nanmax(I)
    if I_max == 0 or np.isnan(I_max):
        return {"fit_success": False, "reason": "invalid normalization"}

    I_norm = I / I_max

    # --- determine direction ---
    if increasing is None:
        # Auto-detect direction
        increasing = np.nanmean(I_norm[-5:]) > np.nanmean(I_norm[:5])
    fit_func = sigmoid_increasing if increasing else sigmoid_decreasing

    # --- compute C bounds from outermost points ---
    n_outer = max(1, int(n_outer_frac * len(I)))
    if increasing:
        outer_values = I[:n_outer]  # last n_outer points
    else:
        outer_values = I[-n_outer:]   # first n_outer points
    outer_mean = np.nanmean(outer_values)
    outer_std = np.nanstd(outer_values)
    C_lower = outer_mean - outer_std
    C_upper = outer_mean + outer_std

    # --- initial guesses ---
    r0_guess = r[np.argmin(np.abs(I_norm - 0.5))]
    s_guess = (np.nanmax(r) - np.nanmin(r)) / 20
    C_guess = outer_mean
    p0 = [r0_guess, s_guess, C_guess]

    bounds = ([np.nanmin(r), 1e-6, C_lower],
              [np.nanmax(r), np.nanmax(r) - np.nanmin(r), C_upper])

    try:
        # --- weighted fit ---
        sigma = None
        if I_std is not None and np.any(np.isfinite(I_std)):
            valid_mask = np.isfinite(I_std)
            sigma = I_std[valid_mask]
            sigma[sigma == 0] = np.min(sigma[sigma > 0])

        popt, _ = curve_fit(
            fit_func,
            r,
            I_norm,
            p0=p0,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=True if sigma is not None else False,
            maxfev=10000
        )

        r0, s, C = popt

        # --- R² ---
        residuals = I_norm - fit_func(r, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((I_norm - np.nanmean(I_norm))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # --- uncertainty at r0 ---
        uncertainty_r0 = np.nan
        if I_std is not None and np.any(np.isfinite(I_std)):
            valid_mask = np.isfinite(I_std)
            uncertainty_r0 = np.interp(r0, r[valid_mask], I_std[valid_mask])

        # --- success criteria ---
        criteria = []

        # 1) uncertainty (optional)
        if not np.isnan(uncertainty_r0):
            criteria.append(uncertainty_r0 < uncertainty_thr)

        # 2) s range (optional)
        if s_min is not None:
            criteria.append(s >= s_min)
        if s_max is not None:
            criteria.append(s <= s_max)

        # 3) plateau_thr (optional)
        if plateau_thr is not None:
            if increasing:
                criteria.append(C > (1 - plateau_thr))
            else:
                criteria.append(C < plateau_thr)

        fit_success = all(criteria)

        return {
            "fit_success": fit_success,
            "direction": "increasing" if increasing else "decreasing",
            "r0": r0,
            "s": s,
            "C": C,
            "r2": r2,
            "uncertainty_r0": uncertainty_r0
        }

    except RuntimeError:
        return {"fit_success": False, "reason": "fit failed"}




def calculate_profile_statistics(intensity: np.ndarray, 
                               relative_dist: np.ndarray, 
                               absolute_dist: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate statistical measures for intensity profile.
    
    Args:
        intensity: Intensity values along the profile
        relative_dist: Relative distances (0 to 1)
        absolute_dist: Absolute distances in μm (optional)
        
    Returns:
        Dictionary containing statistical measures
    """
    stats = {
        'max_intensity': np.max(intensity),
        'min_intensity': np.min(intensity),
        'mean_intensity': np.mean(intensity),
        'std_intensity': np.std(intensity),
        'center_intensity': intensity[0] if len(intensity) > 0 else 0,
        'edge_intensity': intensity[-1] if len(intensity) > 0 else 0,
        'center_to_edge_ratio': intensity[0] / intensity[-1] if len(intensity) > 0 and intensity[-1] > 0 else 0,
    }
    
    if absolute_dist is not None:
        stats['absolute_max_distance'] = float(absolute_dist[-1])
        stats['absolute_center_distance'] = float(absolute_dist[0])
    
    return stats


def find_inflection_points(intensity: np.ndarray, 
                          relative_dist: np.ndarray, 
                          absolute_dist: Optional[np.ndarray] = None) -> Dict[str, Tuple]:
    """
    Find inflection points in the intensity profile.
    
    Args:
        intensity: Intensity values along the profile
        relative_dist: Relative distances (0 to 1)
        absolute_dist: Absolute distances in μm (optional)
        
    Returns:
        Dictionary containing inflection point information
    """
    try:
        # Use existing find_inflection_point function
        inflection_data = find_inflection_point(intensity, relative_dist, sigma=3)
        
        result = {
            'inflection_point': inflection_data.get('inflection', [None, None, None]),
            'minimum': inflection_data.get('minimum', [None, None, None]),
            'maximum': inflection_data.get('maximum', [None, None, None]),
        }
        
        # Convert to absolute distances if available
        if absolute_dist is not None:
            for key in result:
                if result[key][0] is not None:
                    # Interpolate absolute distance
                    abs_val = np.interp(result[key][0], relative_dist, absolute_dist)
                    result[key] = (result[key][0], result[key][1], abs_val)
        
        return result
    except Exception:
        return {'inflection_point': [None, None, None], 
               'minimum': [None, None, None], 
               'maximum': [None, None, None]}


def calculate_thresholds(intensity: np.ndarray, 
                       relative_dist: np.ndarray, 
                       absolute_dist: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
    """
    Calculate various intensity thresholds.
    
    Args:
        intensity: Intensity values along the profile
        relative_dist: Relative distances (0 to 1)
        absolute_dist: Absolute distances in μm (optional)
        
    Returns:
        Dictionary containing threshold information
    """
    max_intensity = np.max(intensity)
    
    thresholds = {
        '50_percent': find_threshold_position(intensity, relative_dist, absolute_dist, 0.5 * max_intensity),
        '25_percent': find_threshold_position(intensity, relative_dist, absolute_dist, 0.25 * max_intensity),
        '75_percent': find_threshold_position(intensity, relative_dist, absolute_dist, 0.75 * max_intensity),
        'center_threshold': find_threshold_position(intensity, relative_dist, absolute_dist, intensity[0] * 0.5),
        'edge_threshold': find_threshold_position(intensity, relative_dist, absolute_dist, intensity[-1] * 2),
    }
    
    return thresholds


def find_threshold_position(intensity: np.ndarray, 
                           relative_dist: np.ndarray, 
                           absolute_dist: Optional[np.ndarray] = None, 
                           threshold_value: float = 0.5) -> Dict[str, Any]:
    """
    Find position where intensity crosses threshold.
    
    Args:
        intensity: Intensity values along the profile
        relative_dist: Relative distances (0 to 1)
        absolute_dist: Absolute distances in μm (optional)
        threshold_value: Threshold intensity value
        
    Returns:
        Dictionary containing threshold position information
    """
    # Find where intensity crosses threshold
    crossings = np.where(np.diff(np.sign(intensity - threshold_value)))[0]
    
    if len(crossings) > 0:
        # Use first crossing
        rel_pos = relative_dist[crossings[0]]
        abs_pos = absolute_dist[crossings[0]] if absolute_dist is not None else None
        return {'relative_position': rel_pos, 'absolute_position': abs_pos, 'found': True}
    else:
        return {'relative_position': None, 'absolute_position': None, 'found': False}


def identify_functional_zones(intensity: np.ndarray, 
                            relative_dist: np.ndarray, 
                            absolute_dist: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Tuple]]:
    """
    Identify functional zones based on intensity patterns.
    
    Args:
        intensity: Intensity values along the profile
        relative_dist: Relative distances (0 to 1)
        absolute_dist: Absolute distances in μm (optional)
        
    Returns:
        Dictionary containing functional zone information
    """
    zones = {}
    
    # High intensity zone (center)
    high_threshold = np.percentile(intensity, 80)
    high_mask = intensity >= high_threshold
    if np.any(high_mask):
        high_start = relative_dist[np.argmax(high_mask)]
        high_end = relative_dist[len(relative_dist) - 1 - np.argmax(high_mask[::-1])]
        zones['high_intensity'] = {
            'relative_range': (high_start, high_end),
            'absolute_range': (absolute_dist[np.argmax(high_mask)], absolute_dist[len(absolute_dist) - 1 - np.argmax(high_mask[::-1])]) if absolute_dist is not None else None
        }
    
    # Low intensity zone (edge)
    low_threshold = np.percentile(intensity, 20)
    low_mask = intensity <= low_threshold
    if np.any(low_mask):
        low_start = relative_dist[np.argmax(low_mask)]
        low_end = relative_dist[len(relative_dist) - 1 - np.argmax(low_mask[::-1])]
        zones['low_intensity'] = {
            'relative_range': (low_start, low_end),
            'absolute_range': (absolute_dist[np.argmax(low_mask)], absolute_dist[len(absolute_dist) - 1 - np.argmax(low_mask[::-1])]) if absolute_dist is not None else None
        }
    
    return zones


def plot_fluorescence_analysis(results: Dict[str, Any], analysis_type: str, 
                             title_prefix: str = "Fluorescence Analysis",
                             savepath: str | None = None) -> None:
    """
    Plot fluorescence analysis results.
    
    Args:
        results: Analysis results dictionary
        analysis_type: Type of analysis performed
        title_prefix: Prefix for plot titles
        savepath: Optional path to save the plot. If None, plot is only displayed.
    """
    n_channels = len(results)
    fig, axes = plt.subplots(1, n_channels, figsize=(5*n_channels, 4))
    if n_channels == 1:
        axes = [axes]
    
    for i, (channel, data) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot intensity profile
        ax.plot(data['relative_distances'], data['intensity_profile'], 
               'b-', linewidth=2, label='Intensity')
        ax.fill_between(data['relative_distances'], 
                      data['intensity_profile'] - data['std_profile'],
                      data['intensity_profile'] + data['std_profile'],
                      alpha=0.3, color='blue')
        
        # Mark inflection points
        if 'inflection_points' in data:
            inf = data['inflection_points']
            if inf['inflection_point'][0] is not None:
                ax.axvline(inf['inflection_point'][0], color='red', linestyle='--', 
                         label=f'Inflection: {inf["inflection_point"][0]:.2f}')
        
        # Mark thresholds
        if 'thresholds' in data:
            thresh = data['thresholds']
            for name, thresh_data in thresh.items():
                if thresh_data['found'] and thresh_data['relative_position'] is not None:
                    ax.axvline(thresh_data['relative_position'], color='green', 
                             linestyle=':', alpha=0.7, label=f'{name}: {thresh_data["relative_position"]:.2f}')
        
        ax.set_xlabel('Relative Distance from Center')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'{title_prefix} - {channel.capitalize()} Channel')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_fluorescence_profile(intensity: np.ndarray,
                               std: np.ndarray,
                               relative_dist: np.ndarray,
                               absolute_dist: Optional[np.ndarray] = None,
                               analysis_type: str = 'comprehensive') -> Dict[str, Any]:
    """
    Comprehensive analysis of a single fluorescence intensity profile.
    
    Args:
        intensity: Intensity values along the profile
        std: Standard deviation values along the profile
        relative_dist: Relative distances (0 to 1)
        absolute_dist: Absolute distances in μm (optional)
        analysis_type: Type of analysis ('comprehensive', 'inflection', 'threshold', 'zones')
        
    Returns:
        Dictionary containing analysis results
    """
    results = {
        'intensity_profile': intensity,
        'std_profile': std,
        'relative_distances': relative_dist,
        'absolute_distances': absolute_dist,
        'statistics': calculate_profile_statistics(intensity, relative_dist, absolute_dist),
    }
    
    # Perform different types of analysis
    if analysis_type in ['comprehensive', 'inflection']:
        results['inflection_points'] = find_inflection_points(
            intensity, relative_dist, absolute_dist
        )
    
    if analysis_type in ['comprehensive', 'threshold']:
        results['thresholds'] = calculate_thresholds(
            intensity, relative_dist, absolute_dist
        )
    
    if analysis_type in ['comprehensive', 'zones']:
        results['functional_zones'] = identify_functional_zones(
            intensity, relative_dist, absolute_dist
        )
    
    return results
