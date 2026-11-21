import numpy as np
import matplotlib.pyplot as plt

MESH_FILE = "habitation_etape5.mesh"

LAMBDA_AIR = 0.25
LAMBDA_RADIATEUR = 1.0

F_RAD = 50.0
U_DIRICHLET = 15.0
G_NEUMANN = 5.0

print("=" * 60)
print("ETAPE 5 - NEUMANN NON HOMOGENE")
print("=" * 60)

# ============================================================
# 1. Lecture du fichier .mesh Medit
# ============================================================

with open(MESH_FILE, "r", encoding="utf-8") as f:
    data = f.read().split()


def position(mot):
    return data.index(mot)


# ------------------------------------------------------------
# Sommets
# ------------------------------------------------------------

p = position("Vertices")
nv = int(data[p + 1])

points = np.zeros((nv, 2))

k = p + 2

for i in range(nv):
    points[i, 0] = float(data[k])
    points[i, 1] = float(data[k + 1])
    k += 3


# ------------------------------------------------------------
# Triangles
# ------------------------------------------------------------

p = position("Triangles")
nt = int(data[p + 1])

triangles = np.zeros((nt, 3), dtype=int)
regions = np.zeros(nt, dtype=int)

k = p + 2

for t in range(nt):
    triangles[t, 0] = int(data[k]) - 1
    triangles[t, 1] = int(data[k + 1]) - 1
    triangles[t, 2] = int(data[k + 2]) - 1
    regions[t] = int(data[k + 3])

    k += 4


print("Fichier :", MESH_FILE)
print("Noeuds :", nv)
print("Triangles :", nt)


# ============================================================
# 2. Identification des regions
# ============================================================

unique_regions, counts = np.unique(regions, return_counts=True)

# La région contenant le plus de triangles est considérée
# comme la région d'air
reg_air = unique_regions[np.argmax(counts)]

# Toutes les autres régions sont considérées comme radiateurs
reg_radiateurs = [r for r in unique_regions if r != reg_air]

print("\nRegions detectees :")
print("Region", reg_air, "-> air")

for r in reg_radiateurs:
    print("Region", r, "-> radiateur")


# ============================================================
# 3. Matrice de rigidite et second membre
# ============================================================

K = np.zeros((nv, nv))
F = np.zeros(nv)

for t in range(nt):

    nodes = triangles[t]
    coords = points[nodes]

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Déterminant géométrique
    detJ = (
        (x2 - x1) * (y3 - y1)
        - (x3 - x1) * (y2 - y1)
    )

    # Aire du triangle
    area = abs(detJ) / 2.0

    if area <= 0:
        raise ValueError(
            f"Triangle de surface nulle : {t}"
        )

    # Coefficients des gradients des fonctions de forme P1
    b = np.array([
        y2 - y3,
        y3 - y1,
        y1 - y2
    ])

    c = np.array([
        x3 - x2,
        x1 - x3,
        x2 - x1
    ])

    grads = np.column_stack((b, c)) / detJ

    # --------------------------------------------------------
    # Propriétés physiques selon la région
    # --------------------------------------------------------

    if regions[t] == reg_air:
        lam = LAMBDA_AIR
        source = 0.0

    else:
        lam = LAMBDA_RADIATEUR
        source = F_RAD

    # --------------------------------------------------------
    # Matrice élémentaire
    # --------------------------------------------------------

    Ke = lam * area * (grads @ grads.T)

    # --------------------------------------------------------
    # Second membre élémentaire
    # --------------------------------------------------------

    Fe = source * area / 3.0 * np.ones(3)

    # --------------------------------------------------------
    # Assemblage
    # --------------------------------------------------------

    for i in range(3):

        I = nodes[i]

        F[I] += Fe[i]

        for j in range(3):

            J = nodes[j]

            K[I, J] += Ke[i, j]


# ============================================================
# 4. Detection des aretes de frontiere
# ============================================================

edge_count = {}

for tri in triangles:

    edges = [
        (tri[0], tri[1]),
        (tri[1], tri[2]),
        (tri[2], tri[0])
    ]

    for i, j in edges:

        edge = tuple(sorted((i, j)))

        edge_count[edge] = edge_count.get(edge, 0) + 1


# Une arête présente une seule fois appartient à la frontière
boundary_edges = [
    edge
    for edge, count in edge_count.items()
    if count == 1
]


# ============================================================
# 5. Identification de F1 et F2
# ============================================================

neumann_edges = []

neumann_nodes = set()
dirichlet_nodes = set()

tol = 1e-8

for i, j in boundary_edges:

    xi, yi = points[i]
    xj, yj = points[j]

    # --------------------------------------------------------
    # F1 : x = 0
    # 2 <= y <= 3.5
    # Neumann non homogène
    # --------------------------------------------------------

    if (
        abs(xi) < tol
        and abs(xj) < tol
        and min(yi, yj) >= 2.0 - tol
        and max(yi, yj) <= 3.5 + tol
    ):

        neumann_edges.append((i, j))

        neumann_nodes.add(i)
        neumann_nodes.add(j)

    # --------------------------------------------------------
    # F2 : x = 6
    # 1 <= y <= 2.5
    # Dirichlet
    # --------------------------------------------------------

    if (
        abs(xi - 6.0) < tol
        and abs(xj - 6.0) < tol
        and min(yi, yj) >= 1.0 - tol
        and max(yi, yj) <= 2.5 + tol
    ):

        dirichlet_nodes.add(i)
        dirichlet_nodes.add(j)


print("\nAretes de frontiere detectees :", len(boundary_edges))
print("Noeuds F1 - Neumann :", len(neumann_nodes))
print("Noeuds F2 - Dirichlet :", len(dirichlet_nodes))
print("Aretes F1 avec flux g :", len(neumann_edges))


# ============================================================
# 6. Terme de Neumann
# ============================================================

# Condition :
#
# -lambda du/dn = g
#
# Avec la convention utilisée ici :
#
# K U = F_volume - F_Neumann

for i, j in neumann_edges:

    xi, yi = points[i]
    xj, yj = points[j]

    # Longueur de l'arête
    longueur = np.sqrt(
        (xj - xi) ** 2
        + (yj - yi) ** 2
    )

    # Contribution P1 du flux de Neumann
    contribution = G_NEUMANN * longueur / 2.0

    F[i] -= contribution
    F[j] -= contribution


# ============================================================
# 7. Condition de Dirichlet exacte
# ============================================================

# Modification du second membre
for node in dirichlet_nodes:

    F -= K[:, node] * U_DIRICHLET


# Modification de la matrice
for node in dirichlet_nodes:

    K[node, :] = 0.0
    K[:, node] = 0.0

    K[node, node] = 1.0

    F[node] = U_DIRICHLET


# ============================================================
# 8. Resolution du systeme
# ============================================================

print("\nResolution du systeme ...")

U = np.linalg.solve(K, F)


# ============================================================
# 8.1 Calcul de l'aire du domaine
# ============================================================

aire = 0.0

for t in range(nt):

    nodes = triangles[t]
    coords = points[nodes]

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    detJ = (
        (x2 - x1) * (y3 - y1)
        - (x3 - x1) * (y2 - y1)
    )

    area = abs(detJ) / 2.0

    aire += area


# ============================================================
# 8.2 Calcul de l'integrale de u
# ============================================================

total = 0.0

for t in range(nt):

    nodes = triangles[t]
    coords = points[nodes]

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    detJ = (
        (x2 - x1) * (y3 - y1)
        - (x3 - x1) * (y2 - y1)
    )

    area = abs(detJ) / 2.0

    # Pour un élément P1 :
    #
    # integral_T(u) =
    # area(T) / 3 * (u1 + u2 + u3)

    total += (
        area / 3.0
        * np.sum(U[nodes])
    )


# ============================================================
# 9. Resultats
# ============================================================

Tmin = np.min(U)
Tmax = np.max(U)

# Température moyenne sur le domaine
Tmoy = total / aire


print("\n" + "=" * 60)
print("RESULTATS ETAPE 5")
print("=" * 60)

print(f"Temperature minimale = {Tmin:.6f} degC")
print(f"Temperature maximale = {Tmax:.6f} degC")
print(f"Aire du domaine = {aire:.12f}")
print(f"Integral de u = {total:.12f}")
print(f"Temperature moyenne = {Tmoy:.6f} degC")


# ============================================================
# 10. Sauvegarde des resultats
# ============================================================

with open(
    "etape5_resultats_corriges.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ETAPE 5 - NEUMANN NON HOMOGENE\n"
    )

    f.write(
        "================================\n"
    )

    f.write(
        f"Maillage : {MESH_FILE}\n"
    )

    f.write(
        f"Noeuds : {nv}\n"
    )

    f.write(
        f"Triangles : {nt}\n"
    )

    f.write(
        f"Lambda air = {LAMBDA_AIR}\n"
    )

    f.write(
        f"Lambda radiateur = {LAMBDA_RADIATEUR}\n"
    )

    f.write(
        f"f radiateurs = {F_RAD} W/m^3\n"
    )

    f.write(
        f"uD = {U_DIRICHLET} degC\n"
    )

    f.write(
        f"g = {G_NEUMANN} W/m^2\n"
    )

    f.write(
        f"Aire = {aire:.12f}\n"
    )

    f.write(
        f"Integral de u = {total:.12f}\n"
    )

    f.write(
        f"Tmin = {Tmin:.6f} degC\n"
    )

    f.write(
        f"Tmax = {Tmax:.6f} degC\n"
    )

    f.write(
        f"Tmoy = {Tmoy:.6f} degC\n"
    )


# ============================================================
# 11. Visualisation
# ============================================================

plt.figure(figsize=(8, 6))

plt.tripcolor(
    points[:, 0],
    points[:, 1],
    triangles,
    U,
    shading="gouraud"
)

plt.colorbar(
    label="Temperature (degC)"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "Etape 5 - Temperature avec Neumann non homogene"
)

plt.tight_layout()

plt.savefig(
    "etape5_temperature.png",
    dpi=300
)

plt.close()


# ============================================================
# FIN
# ============================================================

print("\nResultats sauvegardes.")
print("=" * 60)
