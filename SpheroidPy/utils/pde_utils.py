import numpy as np
import matplotlib.pyplot as plt
from meshpy.triangle import MeshInfo, build
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Berechnung der Laplace-Matrix (Steifigkeitsmatrix) mit Reaktionsterm
def compute_laplace_matrix(points, triangles, reaction_rate, diffusion_rate):
    num_points = len(points)
    A = lil_matrix((num_points, num_points))  # Steifigkeitsmatrix

    for tri in triangles:
        p0, p1, p2 = points[tri]

        # Berechnung der Dreiecksfläche
        area = 0.5 * np.linalg.det(np.array([[1, p0[0], p0[1]],
                                             [1, p1[0], p1[1]],
                                             [1, p2[0], p2[1]]]))

        # Berechnung der Gradienten der Basisfunktionen
        b = np.array([p1[1] - p2[1], p2[1] - p0[1], p0[1] - p1[1]])
        c = np.array([p2[0] - p1[0], p0[0] - p2[0], p1[0] - p0[0]])
        B = np.array([[b[0], b[1], b[2]],
                      [c[0], c[1], c[2]]]) / (2 * area)

        local_A = diffusion_rate * (B.T @ B) * area

        # Reaktionsterm hinzufügen
        for i in range(3):
            local_A[i, i] += reaction_rate * area  # Reaktionsterm nur auf der Diagonalen

        # Update der globalen Steifigkeitsmatrix
        for i in range(3):
            for j in range(3):
                A[tri[i], tri[j]] += local_A[i, j]

    return A


# 3. Berechnung der erweiterten Matrix (Diffusion, Reaktion, Konvektion)
def compute_extended_matrix(points, triangles, reaction_rate, diffusion_constant, velocity_field):
    num_points = len(points)
    A = lil_matrix((num_points, num_points))  # Globale Matrix

    for tri in triangles:
        p0, p1, p2 = points[tri]

        # Berechnung der Dreiecksfläche
        area = 0.5 * np.linalg.det(np.array([[1, p0[0], p0[1]],
                                             [1, p1[0], p1[1]],
                                             [1, p2[0], p2[1]]]))

        # Berechnung der Gradienten der Basisfunktionen
        b = np.array([p1[1] - p2[1], p2[1] - p0[1], p0[1] - p1[1]])
        c = np.array([p2[0] - p1[0], p0[0] - p2[0], p1[0] - p0[0]])
        B = np.array([[b[0], b[1], b[2]],
                      [c[0], c[1], c[2]]]) / (2 * area)

        # 3a. Diffusionsterm
        local_A_diffusion = diffusion_constant * ((B.T @ B) * area)

        # 3b. Konvektionsterm
        # Geschwindigkeitsfeld an den Knoten des Dreiecks interpolieren
        v0, v1, v2 = velocity_field[tri]  # Geschwindigkeit an den Punkten p0, p1, p2
        vx_avg = (v0[0] + v1[0] + v2[0]) / 3  # Mittelwert der x-Komponente
        vy_avg = (v0[1] + v1[1] + v2[1]) / 3  # Mittelwert der y-Komponente

        # Lokaler Konvektionsterm
        grad_phi = B  # .T  # Dimension: 2 x 3
        # print(grad_phi.shape)
        local_A_convection = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                local_A_convection[i, j] = area * (vx_avg * grad_phi[0, j] + vy_avg * grad_phi[1, j]) / 3

        # 3c. Reaktionsterm
        local_A_reaction = np.zeros((3, 3))
        for i in range(3):
            local_A_reaction[i, i] += reaction_rate * area  # Zerfallsrate nur auf der Diagonalen

        # 3d. Gesamte lokale Matrix
        local_A = local_A_diffusion + local_A_reaction  # + local_A_convection

        # Update der globalen Matrix
        for i in range(3):
            for j in range(3):
                A[tri[i], tri[j]] += local_A[i, j]

    return A

# Randbedingungen anwenden (Dirichlet)
def apply_dirichlet_boundary_conditions(A, b, boundary_nodes, boundary_value=10):
    for node in boundary_nodes:
        A[node, :] = 0
        A[node, node] = 1
        b[node] = boundary_value

# Randknoten identifizieren
def get_boundary_nodes(contour_points, mesh_points):
    boundary_nodes = []
    for i, point in enumerate(mesh_points):
        if any(np.allclose(point, contour_point, atol=1e-2) for contour_point in contour_points):
            boundary_nodes.append(i)
    return boundary_nodes
