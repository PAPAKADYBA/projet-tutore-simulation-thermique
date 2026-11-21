#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projet : Simulation numérique du chauffage d'une habitation
Méthode des Éléments Finis P1

La matrice globale est stockée sous forme creuse et la condition de Dirichlet
est imposée exactement par élimination des degrés de liberté prescrits.

Le maillage est créé avec FreeFem++.
Python lit directement le fichier habitation.mesh.
"""

import os
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# ============================================================
# 1. DONNEES PHYSIQUES
# ============================================================

LAMBDA_AIR = 0.25
LAMBDA_ALU = 1.0
U_DIRICHLET = 15.0

# Puissances des radiateurs
F_VALUES = [0, 50, 100, 150, 200, 250]

# Fichier du maillage produit par FreeFem++
MESH_FILE = "habitation.mesh"

# Dossier dans lequel seront enregistrées les figures
FIG_DIR = "figures"


# ============================================================
# 2. LECTURE DU MAILLAGE FREEFEM++
# ============================================================

def read_freefem_mesh(filename):

    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"Le fichier {filename} est introuvable."
        )

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
            and not line.lstrip().startswith("#")
        ]

    def find_block(name):

        for i in range(len(lines)):

            if lines[i].lower() == name.lower():
                return i

        return -1

    iv = find_block("Vertices")
    it = find_block("Triangles")
    ie = find_block("Edges")

    if iv < 0 or it < 0:

        raise ValueError(
            "Les blocs Vertices et Triangles "
            "sont absents du fichier .mesh."
        )

    # --------------------------------------------------------
    # Sommets
    # --------------------------------------------------------

    n_nodes = int(lines[iv + 1])

    nodes = np.zeros(
        (n_nodes, 2),
        dtype=float
    )

    for k in range(n_nodes):

        vals = lines[iv + 2 + k].split()

        nodes[k, 0] = float(vals[0])
        nodes[k, 1] = float(vals[1])

    # --------------------------------------------------------
    # Triangles
    # --------------------------------------------------------

    n_tri = int(lines[it + 1])

    triangles = np.zeros(
        (n_tri, 3),
        dtype=int
    )

    regions = np.ones(
        n_tri,
        dtype=int
    )

    for k in range(n_tri):

        vals = lines[it + 2 + k].split()

        triangles[k, 0] = int(vals[0]) - 1
        triangles[k, 1] = int(vals[1]) - 1
        triangles[k, 2] = int(vals[2]) - 1

        if len(vals) >= 4:
            regions[k] = int(vals[3])

    # --------------------------------------------------------
    # Aretes de frontière
    # --------------------------------------------------------

    edges = np.empty(
        (0, 3),
        dtype=int
    )

    if ie >= 0:

        n_edges = int(lines[ie + 1])

        edges = np.zeros(
            (n_edges, 3),
            dtype=int
        )

        for k in range(n_edges):

            vals = lines[ie + 2 + k].split()

            edges[k, 0] = int(vals[0]) - 1
            edges[k, 1] = int(vals[1]) - 1
            edges[k, 2] = int(vals[2])

    return nodes, triangles, regions, edges


# ============================================================
# 3. AFFICHAGE DU MAILLAGE
# ============================================================

def afficher_maillage(nodes, triangles, edges):

    """
    Affiche le maillage EF-P1.

    Les fenêtres ont le label 2.
    Elles sont affichées avec des pointillés.
    """

    triang = mtri.Triangulation(
        nodes[:, 0],
        nodes[:, 1],
        triangles
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # --------------------------------------------------------
    # Maillage triangulaire
    # --------------------------------------------------------

    ax.triplot(
        triang,
        linewidth=0.35
    )

    # --------------------------------------------------------
    # Fenêtres : label 2
    # --------------------------------------------------------

    for i, j, label in edges:

        if label == 2:

            x1, y1 = nodes[int(i)]
            x2, y2 = nodes[int(j)]

            ax.plot(
                [x1, x2],
                [y1, y2],
                linestyle="--",
                linewidth=2
            )

    # --------------------------------------------------------
    # Mise en forme
    # --------------------------------------------------------

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.set_title(
        "Maillage EF-P1 généré par FreeFem++\n"
        "Fenêtres : pointillés"
    )

    ax.set_aspect("equal")

    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "figure5_maillage_EF_P1.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 4. MATRICE LOCALE P1
# ============================================================

def p1_local_stiffness(coords):

    (x1, y1), (x2, y2), (x3, y3) = coords

    det = (
        (x2 - x1) * (y3 - y1)
        - (x3 - x1) * (y2 - y1)
    )

    area = 0.5 * abs(det)

    if area <= 1e-15:

        raise ValueError(
            "Triangle de surface nulle."
        )

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

    Kloc = (
        np.outer(b, b)
        + np.outer(c, c)
    ) / (4.0 * area)

    return Kloc, area


# ============================================================
# 5. ASSEMBLAGE DE LA MATRICE ET DU SECOND MEMBRE
# ============================================================

def assemble(
    nodes,
    triangles,
    regions,
    region_air,
    active_regions,
    f_rad
):

    n = len(nodes)

    rows = []
    cols = []
    vals = []

    F = np.zeros(n)

    for t, tri in enumerate(triangles):

        coords = nodes[tri]

        Kloc, area = p1_local_stiffness(coords)

        region = regions[t]

        # ----------------------------------------------------
        # Air
        # ----------------------------------------------------

        if region == region_air:

            lam = LAMBDA_AIR
            fval = 0.0

        # ----------------------------------------------------
        # Radiateur
        # ----------------------------------------------------

        else:

            lam = LAMBDA_ALU

            if int(region) in active_regions:
                fval = f_rad
            else:
                fval = 0.0

        # Matrice locale
        Kloc = lam * Kloc

        # Assemblage de la matrice globale
        for a in range(3):

            for b in range(3):

                rows.append(tri[a])
                cols.append(tri[b])
                vals.append(Kloc[a, b])

        # Second membre local
        Floc = fval * area / 3.0

        for a in range(3):

            F[tri[a]] += Floc

    K = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(n, n)
    )

    return K, F


# ============================================================
# 6. NOEUDS DE DIRICHLET
# ============================================================

def get_dirichlet_nodes(edges):

    """
    Les fenêtres ont le label 2.
    """

    dirichlet = set()

    for i, j, label in edges:

        if label == 2:

            dirichlet.add(int(i))
            dirichlet.add(int(j))

    return np.array(
        sorted(dirichlet),
        dtype=int
    )


# ============================================================
# 7. IMPOSITION DE DIRICHLET
# ============================================================

def apply_dirichlet_exact(K, F, dnodes, uD):
    """Imposition exacte de Dirichlet par élimination des ddl prescrits."""
    K = K.tocsr(copy=True)
    F = F.copy()

    if len(dnodes) == 0:
        return K, F

    # Pour les inconnues libres L :
    # K_LL U_L = F_L - K_LD U_D.
    F -= np.asarray(K[:, dnodes].sum(axis=1)).ravel() * uD

    K = K.tolil()
    for k in dnodes:
        K[k, :] = 0.0
        K[:, k] = 0.0
        K[k, k] = 1.0
        F[k] = uD

    return K.tocsr(), F


def area_weighted_mean(nodes, triangles, u):
    total = 0.0
    area_total = 0.0
    for tri in triangles:
        _, area = p1_local_stiffness(nodes[tri])
        total += area * np.mean(u[tri])
        area_total += area
    return total / area_total


# ============================================================
# 8. RESOLUTION
# ============================================================

def solve_configuration(
    nodes,
    triangles,
    regions,
    edges,
    region_air,
    radiator_regions,
    active_numbers,
    f_rad
):

    # Regions des radiateurs allumés
    active_regions = {
        radiator_regions[k - 1]
        for k in active_numbers
    }

    # Assemblage
    K, F = assemble(
        nodes,
        triangles,
        regions,
        region_air,
        active_regions,
        f_rad
    )

    # Noeuds des fenêtres
    dnodes = get_dirichlet_nodes(
        edges
    )

    # Dirichlet
    Kd, Fd = apply_dirichlet_exact(
        K, F, dnodes, U_DIRICHLET
    )

    # Résolution
    u = spla.spsolve(
        Kd.tocsc(),
        Fd
    )

    return (
        u,
        float(u.min()),
        float(u.max()),
        float(area_weighted_mean(nodes, triangles, u))
    )


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

def main():

    # Création du dossier figures
    os.makedirs(
        FIG_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Lecture du maillage
    # --------------------------------------------------------

    nodes, triangles, regions, edges = read_freefem_mesh(
        MESH_FILE
    )

    print("==============================================")
    print(" ETAPE 4 - REGLAGE DES RADIATEURS")
    print("==============================================")

    print(
        f"Fichier : {MESH_FILE}"
    )

    print(
        f"Noeuds : {len(nodes)}"
    )

    print(
        f"Triangles : {len(triangles)}"
    )

    print(
        f"Lambda air = {LAMBDA_AIR}"
    )

    print(
        f"Lambda radiateur = {LAMBDA_ALU}"
    )

    print(
        f"uD = {U_DIRICHLET} degC"
    )

    # --------------------------------------------------------
    # Figure 5
    # --------------------------------------------------------

    print("\nGeneration de la Figure 5...")

    afficher_maillage(
        nodes,
        triangles,
        edges
    )

    print(
        "Figure 5 : maillage + fenêtres en pointillés"
    )

    # --------------------------------------------------------
    # Recherche des régions
    # --------------------------------------------------------

    unique_regions, counts = np.unique(
        regions,
        return_counts=True
    )

    # La région la plus grande est considérée comme l'air
    region_air = int(
        unique_regions[
            np.argmax(counts)
        ]
    )

    # Les autres régions sont les radiateurs
    radiator_regions = [
        int(r)
        for r in unique_regions
        if int(r) != region_air
    ]

    if len(radiator_regions) == 0:

        raise ValueError(
            "Aucun radiateur détecté."
        )

    print("\nRegions détectées :")

    print(
        f"  Region {region_air} -> air"
    )

    for k, r in enumerate(
        radiator_regions,
        start=1
    ):

        index = np.where(
            unique_regions == r
        )[0][0]

        ntri = int(
            counts[index]
        )

        print(
            f"  Radiateur {k} -> "
            f"region {r} "
            f"({ntri} triangles)"
        )

    # ========================================================
    # A. EFFET DE LA PUISSANCE
    # ========================================================

    all_numbers = tuple(
        range(
            1,
            len(radiator_regions) + 1
        )
    )

    rows_power = []

    print("\nEffet de la puissance :")

    for f_rad in F_VALUES:

        u, umin, umax, umean = solve_configuration(
            nodes,
            triangles,
            regions,
            edges,
            region_air,
            radiator_regions,
            all_numbers,
            f_rad
        )

        rows_power.append([
            f_rad,
            umin,
            umax,
            umean
        ])

        print(
            f"f = {f_rad:7.2f} "
            f"-> min = {umin:8.3f}, "
            f"max = {umax:8.3f}, "
            f"moyenne = {umean:8.3f}"
        )

    # --------------------------------------------------------
    # Enregistrement CSV
    # --------------------------------------------------------

    power_table = np.array(
        rows_power
    )

    np.savetxt(
        os.path.join(
            FIG_DIR,
            "etape4_puissance.csv"
        ),
        power_table,
        delimiter=",",
        header=(
            "f_W_m3,"
            "temperature_min,"
            "temperature_max,"
            "temperature_moyenne"
        ),
        comments=""
    )

    # --------------------------------------------------------
    # Graphique puissance / température
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.5, 5)
    )

    ax.plot(
        power_table[:, 0],
        power_table[:, 1],
        marker="o",
        label="Temperature minimale"
    )

    ax.plot(
        power_table[:, 0],
        power_table[:, 2],
        marker="o",
        label="Temperature maximale"
    )

    ax.plot(
        power_table[:, 0],
        power_table[:, 3],
        marker="o",
        label="Temperature moyenne"
    )

    ax.set_xlabel(
        "Puissance volumique f (W/m$^3$)"
    )

    ax.set_ylabel(
        "Temperature (degC)"
    )

    ax.set_title(
        "Etape 4 - Effet de la puissance"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "etape4_puissance.png"
        ),
        dpi=200
    )

    plt.close(fig)

    # ========================================================
    # B. ALLUMAGE / EXTINCTION
    # ========================================================

    combinations = []

    # Un seul radiateur
    for k in range(
        1,
        len(radiator_regions) + 1
    ):

        combinations.append(
            (k,)
        )

    # Deux radiateurs
    if len(radiator_regions) >= 2:

        for i in range(
            1,
            len(radiator_regions) + 1
        ):

            for j in range(
                i + 1,
                len(radiator_regions) + 1
            ):

                combinations.append(
                    (i, j)
                )

    # Tous les radiateurs
    if len(radiator_regions) >= 3:

        combinations.append(
            all_numbers
        )

    # Puissance maximale
    f_test = F_VALUES[-1]

    rows_comb = []

    print(
        "\nEffet de l'allumage / extinction :"
    )

    for active in combinations:

        u, umin, umax, umean = solve_configuration(
            nodes,
            triangles,
            regions,
            edges,
            region_air,
            radiator_regions,
            active,
            f_test
        )

        label = " et ".join(
            str(k)
            for k in active
        )

        rows_comb.append([
            label,
            f_test,
            umin,
            umax,
            umean
        ])

        print(
            f"Radiateur(s) {label:>8} "
            f"-> min = {umin:8.3f}, "
            f"max = {umax:8.3f}, "
            f"moyenne = {umean:8.3f}"
        )

    # --------------------------------------------------------
    # CSV allumage
    # --------------------------------------------------------

    with open(
        os.path.join(
            FIG_DIR,
            "etape4_allumage.csv"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "radiateurs_allumes,"
            "f_W_m3,"
            "temperature_min,"
            "temperature_max,"
            "temperature_moyenne\n"
        )

        for row in rows_comb:

            f.write(
                f"{row[0]},"
                f"{row[1]},"
                f"{row[2]:.8f},"
                f"{row[3]:.8f},"
                f"{row[4]:.8f}\n"
            )

    # --------------------------------------------------------
    # Graphique allumage
    # --------------------------------------------------------

    labels = [
        r[0]
        for r in rows_comb
    ]

    x = np.arange(
        len(labels)
    )

    mins = [
        r[2]
        for r in rows_comb
    ]

    maxs = [
        r[3]
        for r in rows_comb
    ]

    means = [
        r[4]
        for r in rows_comb
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        x,
        mins,
        marker="o",
        label="Temperature minimale"
    )

    ax.plot(
        x,
        maxs,
        marker="o",
        label="Temperature maximale"
    )

    ax.plot(
        x,
        means,
        marker="o",
        label="Temperature moyenne"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_xlabel(
        "Radiateurs allumes"
    )

    ax.set_ylabel(
        "Temperature (degC)"
    )

    ax.set_title(
        f"Etape 4 - Allumage/extinction "
        f"(f = {f_test:g} W/m$^3$)"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "etape4_allumage.png"
        ),
        dpi=200
    )

    plt.close(fig)

    # ========================================================
    # C. CHAMP DE TEMPERATURE
    # ========================================================

    triang = mtri.Triangulation(
        nodes[:, 0],
        nodes[:, 1],
        triangles
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 6)
    )

    levels = np.linspace(
        u.min(),
        u.max(),
        25
    )

    cf = ax.tricontourf(
        triang,
        u,
        levels=levels,
        cmap="inferno"
    )

    ax.tricontour(
        triang,
        u,
        levels=10,
        colors="k",
        linewidths=0.25,
        alpha=0.5
    )

    cbar = plt.colorbar(
        cf,
        ax=ax
    )

    cbar.set_label(
        "Temperature (deg C)"
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        "Champ de temperature stationnaire"
    )

    ax.set_xlabel(
        "x (m)"
    )

    ax.set_ylabel(
        "y (m)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "temperature.png"
        ),
        dpi=200
    )

    plt.close(fig)

    # ========================================================
    # D. COUPE 1D
    # ========================================================

    y_cut = 2.7

    mask = np.isclose(
        nodes[:, 1],
        y_cut,
        atol=1e-9
    )

    xs_cut = nodes[mask, 0]
    u_cut = u[mask]

    if len(xs_cut) > 0:

        order = np.argsort(
            xs_cut
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.plot(
            xs_cut[order],
            u_cut[order],
            color="darkred",
            linewidth=1.8
        )

        ax.axvspan(
            0.3,
            1.3,
            color="firebrick",
            alpha=0.2,
            label="Radiateur R1"
        )

        ax.set_xlabel(
            "x (m)"
        )

        ax.set_ylabel(
            "Temperature (deg C)"
        )

        ax.set_title(
            f"Coupe 1D de la temperature "
            f"le long de y = {y_cut} m"
        )

        ax.legend()

        ax.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIG_DIR,
                "coupe1d.png"
            ),
            dpi=200
        )

        plt.close(fig)

    else:

        print(
            "Aucun noeud exactement sur "
            f"y = {y_cut} m."
        )

    # ========================================================
    # E. FICHIER JSON
    # ========================================================

    recap = {

        "lambda_air": LAMBDA_AIR,

        "lambda_radiateur": LAMBDA_ALU,

        "uD": U_DIRICHLET,

        "f_values": F_VALUES,

        "region_air": region_air,

        "regions_radiateurs": radiator_regions,

        "table_puissance": rows_power,

        "table_allumage": rows_comb
    }

    with open(
        os.path.join(
            FIG_DIR,
            "etape4_resultats.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            recap,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # F. AFFICHAGE FINAL
    # ========================================================

    print("\n==============================================")
    print(" FICHIERS GENERES")
    print("==============================================")

    print(
        "  - figure5_maillage_EF_P1.png"
    )

    print(
        "  - etape4_puissance.csv"
    )

    print(
        "  - etape4_puissance.png"
    )

    print(
        "  - etape4_allumage.csv"
    )

    print(
        "  - etape4_allumage.png"
    )

    print(
        "  - etape4_resultats.json"
    )

    print(
        "  - temperature.png"
    )

    if os.path.isfile(
        os.path.join(
            FIG_DIR,
            "coupe1d.png"
        )
    ):

        print(
            "  - coupe1d.png"
        )

    print("\nExecution terminee.")


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    main()

