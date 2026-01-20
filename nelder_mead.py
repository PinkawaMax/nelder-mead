# -*- coding: utf-8 -*-
"""SIMPLEX Minimumssuche (Nelder–Mead / Downhill-Simplex) in 2D.

Beispiel
--------

>>> x_min, f_min, N = simplex(himmelblau, [0, 0], 1000, 1e-12)

Optional kann eine History für Visualisierung zurückgegeben werden:

>>> x_min, f_min, N, hist = simplex(himmelblau, [0, 0], 1000, 1e-12, return_history=True)

"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Tuple, Optional

Array = np.ndarray

# ==================================================
# Parameter (empfohlene Standardwerte)
# ==================================================
alpha_ = 1.0   # Spiegelung (reflection)
beta_ = 0.5    # Kontraktion (contraction)
gamma_ = 2.0   # Expansion (expansion)
lambda_ = 0.1  # Größe Startsimplex
sigma_ = 0.5   # Kompression/Shrink


# ==================================================
# Hilfsfunktionen
# ==================================================

def _centroid_excluding_worst(x_sorted: Array) -> Array:
    """Zentroid aller Punkte außer dem schlechtesten (2D, 3 Punkte)."""
    # bei 3 Punkten ist das der Mittelwert aus best + mid
    return 0.5 * (x_sorted[0] + x_sorted[1])


def spiegeln(x_worst: Array, x_centroid: Array) -> Array:
    """Spiegelung des schlechtesten Punktes am Zentroid."""
    return x_centroid + alpha_ * (x_centroid - x_worst)


def expansion(x_reflected: Array, x_centroid: Array) -> Array:
    """Expansion von der Mitte in Richtung des reflektierten Punktes."""
    return x_centroid + gamma_ * (x_reflected - x_centroid)


def kontraktion_outside(x_reflected: Array, x_centroid: Array) -> Array:
    """Outside-Contraction (wenn reflektierter Punkt besser als worst, aber nicht gut genug)."""
    return x_centroid + beta_ * (x_reflected - x_centroid)


def kontraktion_inside(x_worst: Array, x_centroid: Array) -> Array:
    """Inside-Contraction (wenn reflektierter Punkt nicht besser ist als worst)."""
    return x_centroid + beta_ * (x_worst - x_centroid)


def kompression(x_sorted: Array) -> Array:
    """Shrink: ziehe alle Punkte (außer best) in Richtung best."""
    best = x_sorted[0]
    x_sorted[1] = best + sigma_ * (x_sorted[1] - best)
    x_sorted[2] = best + sigma_ * (x_sorted[2] - best)
    return x_sorted


# ==================================================
# Algorithmus
# ==================================================

def simplex(
    fhandle: Callable[[Array], float],
    x_start: Array,
    N_max: int,
    p: float,
    return_history: bool = False,
) -> Tuple:
    """Minimiert fhandle(x) mit Nelder–Mead (2D).

    Parameter
    ---------
    fhandle : callable
        Zu minimierende Funktion f(x) mit x.shape == (2,)
    x_start : array-like
        Startpunkt (x, y)
    N_max : int
        Maximale Iterationen
    p : float
        Toleranz (hier: Spread der Funktionswerte)
    return_history : bool
        Wenn True, wird eine Liste der Simplex-Vertices pro Iteration mitgegeben

    Returns
    -------
    Wenn return_history == False (Default):
        (x_min, f_min, N)
    Wenn return_history == True:
        (x_min, f_min, N, history)
        wobei history eine Liste von Vertices (shape (3,2)) je Iteration ist.
    """

    x_start = np.asarray(x_start, dtype=float).reshape(2,)

    # Startsimplex (2D): x0, x0+step*e1, x0+step*e2
    x = np.array(
        [
            x_start,
            x_start + np.array([lambda_, 0.0]),
            x_start + np.array([0.0, lambda_]),
        ],
        dtype=float,
    )

    def eval_f(vertices: Array) -> Array:
        return np.array([fhandle(vertices[i]) for i in range(3)], dtype=float)

    f = eval_f(x)

    history: Optional[List[Array]] = [] if return_history else None
    if return_history:
        history.append(x.copy())

    for N in range(int(N_max)):
        # Sortieren: f klein -> groß
        idx = np.argsort(f)
        x = x[idx]
        f = f[idx]

        f_best, f_mid, f_worst = f[0], f[1], f[2]

        # Abbruchbedingung: Spread der Funktionswerte
        if np.max(np.abs(f - f_best)) < p:
            if return_history:
                return x[0].copy(), float(f_best), N, history
            return x[0].copy(), float(f_best), N

        centroid = _centroid_excluding_worst(x)

        # 1) Spiegelung
        xr = spiegeln(x_worst=x[2], x_centroid=centroid)
        fr = float(fhandle(xr))

        if fr < f_best:
            # 2) Expansion
            xe = expansion(x_reflected=xr, x_centroid=centroid)
            fe = float(fhandle(xe))

            if fe < fr:
                x[2], f[2] = xe, fe
            else:
                x[2], f[2] = xr, fr

        elif fr < f_mid:
            # akzeptiere Spiegelung
            x[2], f[2] = xr, fr

        else:
            # 3) Kontraktion
            if fr < f_worst:
                # outside contraction
                xc = kontraktion_outside(x_reflected=xr, x_centroid=centroid)
            else:
                # inside contraction
                xc = kontraktion_inside(x_worst=x[2], x_centroid=centroid)

            fc = float(fhandle(xc))

            if fc < f_worst:
                x[2], f[2] = xc, fc
            else:
                # 4) Kompression / Shrink
                x = kompression(x)
                f = eval_f(x)

        if return_history:
            history.append(x.copy())

    # max iteration reached
    idx = np.argsort(f)
    x = x[idx]
    f = f[idx]

    if return_history:
        return x[0].copy(), float(f[0]), int(N_max), history
    return x[0].copy(), float(f[0]), int(N_max)


# ==================================================
# Demo / schneller Selbsttest
# ==================================================

def himmelblau(x: Array) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (x1 * x1 + x2 - 11.0) ** 2 + (x1 + x2 * x2 - 7.0) ** 2


if __name__ == "__main__":
    x_min, f_min, N = simplex(himmelblau, [0, 0], N_max=2000, p=1e-12, return_history=False)
    print("x_min:", x_min)
    print("f_min:", f_min)
    print("N:", N)
