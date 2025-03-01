import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


def find_inflection_point(y_array, x_array=None, sigma=None):
    """
    Find the (global) maximum, minimum and inflection point of a curve given x and y arrays.

    Parameters:
    - x_array (numpy array): The x-values (e.g., normalized distance).
    - y_array (numpy array): The y-values (e.g., normalized intensity).

    Returns:
    - ...
    """

    if sigma is not None:
        # Smooth the data using a Gaussian filter
        y_array = gaussian_filter1d(y_array, sigma=sigma)

    if x_array is None:
        x_array = np.linspace(0, 1, len(y_array))

    # Calculate the first derivative
    dy_dx_array = np.gradient(y_array, x_array)

    #plt.plot(x_array,np.abs(dy_dx_array))
    #plt.show()

    # Global maximum
    max_index = np.argmax(y_array)
    x_global_max = x_array[max_index]
    y_global_max = y_array[max_index]

    # Global maximum
    min_index = np.argmin(y_array)
    x_global_min = x_array[min_index]
    y_global_min = y_array[min_index]

    # Global inflection point (where second derivative is closest to zero)
    inflection_index = np.argmax(np.abs(dy_dx_array))
    x_global_inflection = x_array[inflection_index]
    y_global_inflection = y_array[inflection_index]

    try:
        # Find the global minimum value
        inflection_min_index = argrelextrema(np.abs(dy_dx_array), np.less)[0][-1]
        x_inflection_min = x_array[inflection_min_index]
        y_inflection_min = y_array[inflection_min_index]
    except:
        inflection_min_index,x_inflection_min,y_inflection_min = None, None, None

    return {'maximum': (x_global_max, y_global_max, max_index), 'minimum': (x_global_min, y_global_min, min_index), 'inflection': (x_global_inflection, y_global_inflection, inflection_index), 'inflection_min': (x_inflection_min, y_inflection_min, inflection_min_index)}


def touches_border(contour: np.ndarray, x_max, y_max, x_min=0, y_min=0) -> bool:

    # Überprüfen, ob der Array den Rand berührt
    x_touch = np.any((contour[:, 0] <= x_min) | (contour[:, 0] >= x_max))
    y_touch = np.any((contour[:, 1] <= y_min) | (contour[:, 1] >= y_max))

    return x_touch or y_touch

def remove_border_points(contour: np.ndarray, x_max, y_max, x_min=0, y_min=0) -> np.ndarray:
    # Überprüfen, welche Punkte den Rand berühren
    border_touching = (contour[:, 0] <= x_min) | (contour[:, 0] >= x_max) | (contour[:, 1] <= y_min) | (contour[:, 1] >= y_max)

    # Punkte filtern, die den Rand nicht berühren
    filtered_contour = contour[border_touching==False]
    #print(contour[border_touching==True][::70])

    return filtered_contour

import numpy as np
from skimage.measure import EllipseModel

def fit_ellipse(contour_points: np.ndarray) -> EllipseModel:
    # Fitten der Ellipse
    ellipse_model = EllipseModel()
    success = ellipse_model.estimate(contour_points)

    # Berechnung der Fläche der Ellipse, wenn das Fitting erfolgreich war
    if success:
        xc, yc, a, b, theta = ellipse_model.params  # xc, yc: Mittelpunkt; a, b: Halbachsen
        ellipse_area = np.pi * a * b
        center = (xc, yc)
        effective_radius = np.sqrt(a * b)
        #print(ellipse_model, 'a')

        # Für Rücgabe der kontur
        t = np.linspace(0, 2 * np.pi, len(contour_points))
        # Parameterform der Ellipse
        x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
        fitted_contour = np.array(list(zip(x, y)), dtype=np.float32)

        return center, ellipse_area, effective_radius, fitted_contour
    else:
        print("Das Modell konnte keine Ellipse anpassen.")

def add_fitted_contour(contour_original: np.ndarray, contour_fitted: np.ndarray, x_max, y_max, x_min=0, y_min=0):


    # Kombinierte Kontur initialisieren
    contour_combined = []

    # Variable zur Steuerung, ob wir gerade außerhalb des Randbereichs sind
    outside = False
    fitted_bool_array = [False, False, False, False]

    # Schritt 1: Gehe jeden Punkt der Originalkontur durch
    for i, point in enumerate(contour_original):
        added_extrapolated = False

        # Fall 1: Punkt liegt rechts außerhalb des Bereichs (x > x_max)
        if point[0] >= x_max and fitted_bool_array[0] == False:
            # Füge die extrapolierten Punkte mit x > x_max ein
            for fit_point in contour_fitted[np.argsort(contour_fitted[:, 1])[::-1]]:
                if fit_point[0] >= x_max:
                    contour_combined.append(fit_point)
                    added_extrapolated = True
                fitted_bool_array[0] = True

        # Fall 2: Punkt liegt links außerhalb des Bereichs (x < x_min)
        elif point[0] <= x_min and fitted_bool_array[1] == False:
            # Füge die extrapolierten Punkte mit x < x_min ein
            for fit_point in contour_fitted[np.argsort(contour_fitted[:, 1])]:
                if fit_point[0] <= x_min:
                    contour_combined.append(fit_point)
                    added_extrapolated = True
                fitted_bool_array[1] = True

        # Fall 3: Punkt liegt oberhalb des Bereichs (y < y_min)
        elif point[1] <= y_min and fitted_bool_array[2] == False:
            # Füge die extrapolierten Punkte mit y < y_min ein
            for fit_point in contour_fitted[np.argsort(contour_fitted[:, 0])[::-1]]:
                if fit_point[1] <= y_min:
                    contour_combined.append(fit_point)
                    added_extrapolated = True
                fitted_bool_array[2] = True

        # Fall 4: Punkt liegt unterhalb des Bereichs (y > y_max)
        elif point[1] >= y_max and fitted_bool_array[3] == False:
            # _fitted_points = contour_fitted[contour_fitted[:,0]>y_max]
            #print(point, fit_point, type(fit_point))
            # Füge die extrapolierten Punkte mit y > y_max ein
            for fit_point in contour_fitted[np.argsort(contour_fitted[:, 0])]:
                if fit_point[1] >= y_max:
                    #print(point, fit_point, type(fit_point))#, np.array(int(fit_point[0]), int(fit_point[1])))
                    contour_combined.append(fit_point.astype(int))
                    added_extrapolated = True
            fitted_bool_array[3] = True

        # Füge den Originalpunkt hinzu, falls kein extrapolierter Punkt eingefügt wurde
        if not added_extrapolated:
            contour_combined.append(point)

    # Konvertiere die Liste in ein Array
    contour_combined = np.array(contour_combined, dtype=np.float32)
    return contour_combined

    # Ausgabe der kombinierten Kontur
    # print("Kombinierte Kontur:", contour_combined)
    # plt.plot(contour_combined[:, 0], contour_combined[:, 1], color='r')