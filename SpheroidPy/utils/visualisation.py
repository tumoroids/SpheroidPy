''' Excel '''
def interpolate_color(value, min_value, max_value, start_color, end_color):
    # Farbbereiche in RGB dekodieren
    start_r = int(start_color[0:2], 16)
    start_g = int(start_color[2:4], 16)
    start_b = int(start_color[4:6], 16)

    end_r = int(end_color[0:2], 16)
    end_g = int(end_color[2:4], 16)
    end_b = int(end_color[4:6], 16)

    # Berechne den Farbwert basierend auf dem Verhältnis
    if max_value > min_value:
        ratio = (value - min_value) / (max_value - min_value)
    else:
        ratio = 0  # Vermeide Division durch 0

    # Interpolierte RGB-Werte
    r = int(start_r + (end_r - start_r) * ratio)
    g = int(start_g + (end_g - start_g) * ratio)
    b = int(start_b + (end_b - start_b) * ratio)

    # Rückgabe des Farbwertes als Hex-String
    return f'{r:02X}{g:02X}{b:02X}'

''' SpheroidImage'''
def plot_mesh(points, triangles, savepath: str | None = None):
    import matplotlib.pyplot as plt
    """
    Plottet ein Mesh.
    
    Args:
        points: Mesh vertex coordinates
        triangles: Mesh connectivity
        savepath: Optional path to save the plot. If None, plot is only displayed.
    """
    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], triangles, color='blue', linewidth=0.5)
    #plt.plot(points[:, 0], points[:, 1], 'o', color='red')
    plt.title('ContourMesh', fontweight='bold')
    plt.xlabel('x-position [µm]')
    plt.ylabel('y-position [µm]')
    plt.axis('equal')
    # Flip y to have origin at top-left if input points use image coordinates
    if points.shape[0] > 0:
        ymin, ymax = points[:, 1].min(), points[:, 1].max()
        if ymax > ymin:
            plt.ylim(ymax, ymin)
    plt.grid()
    if savepath is not None:
        plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()


