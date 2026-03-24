"""Color palette utilities for assigning distinct colors to collections."""

import itertools
import numpy as np

# --------------------------------------------------
# CARTO sequential palettes with simple English names
# --------------------------------------------------

CARTO_SEQUENTIAL = {
    "Gold":     ["#7f4f00", "#a66200", "#cc8400", "#e8a823", "#f5c84a", "#fde480", "#fff3b5"],
    "Red":      ["#7f0000", "#b31b1b", "#d94801", "#f16913", "#fd8d3c", "#fdae6b", "#fee6ce"],
    "Burgundy": ["#6e003b", "#8f1552", "#af3269", "#c95486", "#df77a4", "#f19ac3", "#ffbcd9"],
    "Purple":   ["#3f007d", "#54278f", "#6a51a3", "#807dba", "#9e9ac8", "#bcbddc", "#dadaeb"],
    "Blue":     ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#deebf7"],
    "Teal":     ["#004c4c", "#006d6d", "#238b8b", "#41a9a9", "#74c8c8", "#a1e3e3", "#c8f5f5"],
    "Green":    ["#00441b", "#006d2c", "#238b45", "#41ab5d", "#74c476", "#a1d99b", "#c7e9c0"],
    "Olive":    ["#3d3d00", "#5c5c00", "#7a7a00", "#9b9b1a", "#b8b84d", "#d4d480", "#ededb5"]
}

# --------------------------------------------------
# Utility: convert HEX → Lab (approx via sRGB → XYZ → Lab)
# --------------------------------------------------
def hex_to_lab(hex_color):
    """Convert HEX color to Lab color space."""
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    
    # linearize
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    
    r, g, b = lin(r), lin(g), lin(b)
    
    # sRGB → XYZ (D65)
    X = r*0.4124564 + g*0.3575761 + b*0.1804375
    Y = r*0.2126729 + g*0.7151522 + b*0.0721750
    Z = r*0.0193339 + g*0.1191920 + b*0.9503041
    
    # XYZ → Lab
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    
    def f(t):
        return t**(1/3) if t > 0.008856 else 7.787*t + 16/116
    
    L = 116 * f(Y/Yn) - 16
    a = 500 * (f(X/Xn) - f(Y/Yn))
    b = 200 * (f(Y/Yn) - f(Z/Zn))
    
    return np.array([L, a, b])

# --------------------------------------------------
# Compute representative color (darkest = first)
# --------------------------------------------------
REP_COLORS = {name: hex_to_lab(colors[0]) for name, colors in CARTO_SEQUENTIAL.items()}

# --------------------------------------------------
# Main function: pick k maximally distinct palettes
# Uses MaxMin sampling in Lab space.
# --------------------------------------------------
def pick_maximally_distinct_palettes(k, candidates=None):
    """
    Pick k maximally distinct color palettes from CARTO_SEQUENTIAL.
    
    Uses MaxMin sampling in Lab color space to ensure visual distinctness.
    
    Args:
        k: Number of palettes to select
        candidates: Optional list of palette names to choose from.
            If None, uses all available palettes.
    
    Returns:
        List of palette names (strings) that are maximally distinct.
    """
    if candidates is None:
        candidates = list(CARTO_SEQUENTIAL.keys())
    
    labs = {name: REP_COLORS[name] for name in candidates}
    
    # Start with the palette that is most "central" (farthest from others on average)
    keys = list(labs.keys())
    D = np.zeros((len(keys), len(keys)))
    
    # pairwise distances
    for i, j in itertools.product(range(len(keys)), repeat=2):
        if i < j:
            D[i, j] = np.linalg.norm(labs[keys[i]] - labs[keys[j]])
            D[j, i] = D[i, j]
    
    # seed = most distant from others (max average distance)
    avg_dist = D.mean(axis=1)
    seed_idx = np.argmax(avg_dist)
    selected = [keys[seed_idx]]
    
    # MaxMin selection
    remaining = set(keys) - set(selected)
    while len(selected) < k and remaining:
        best = None
        best_dist = -1
        for cand in remaining:
            # distance to closest selected
            dist = min(
                np.linalg.norm(labs[cand] - labs[s]) for s in selected
            )
            if dist > best_dist:
                best_dist = dist
                best = cand
        selected.append(best)
        remaining.remove(best)
    
    return selected



