#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:21:41 2026

@author: max
"""
# visualize_simplex.py
# Visualisierung fuer Nelder–Mead / Downhill Simplex (2D)
# Kompatibel mit: CP1_U3_simplex_skeleton.py (simplex(..., return_history=True))

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Array = np.ndarray


def _grid_eval(f, xlim, ylim, ngrid: int = 250) -> tuple[Array, Array, Array]:
    """Evaluates f on a 2D grid for contour plots."""
    xs = np.linspace(xlim[0], xlim[1], ngrid)
    ys = np.linspace(ylim[0], ylim[1], ngrid)
    X, Y = np.meshgrid(xs, ys)

    # Robust evaluation (works for f(x: ndarray)->float)
    Z = np.empty_like(X, dtype=float)
    for i in range(ngrid):
        for j in range(ngrid):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]], dtype=float))
    return X, Y, Z


def plot_contours(ax, f, xlim, ylim, ngrid: int = 250, levels: int = 40):
    """Draw contour lines of f on ax."""
    X, Y, Z = _grid_eval(f, xlim, ylim, ngrid=ngrid)
    cs = ax.contour(X, Y, Z, levels=levels)
    ax.clabel(cs, inline=True, fontsize=7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return X, Y, Z


def _close_triangle(vertices: Array) -> Array:
    """vertices shape (3,2) -> returns (4,2) with first point repeated at end."""
    return np.vstack([vertices, vertices[0]])


def plot_simplex(ax, vertices: Array, linewidth: float = 1.0, marker: str = "o", alpha: float = 1.0):
    """Plot one simplex triangle."""
    tri = _close_triangle(vertices)
    ax.plot(tri[:, 0], tri[:, 1], linewidth=linewidth, marker=marker, alpha=alpha)


def plot_history_static(
    f,
    history: list[Array],
    xlim=(-6, 6),
    ylim=(-6, 6),
    ngrid: int = 250,
    levels: int = 40,
    every: int = 10,
    show_best_path: bool = True,
    savepath: str | None = None,
):
    """
    Static plot:
      - contours of f
      - simplex triangles every N iterations
      - last simplex highlighted
      - optional best-point path
    """
    fig, ax = plt.subplots()

    plot_contours(ax, f, xlim, ylim, ngrid=ngrid, levels=levels)
    ax.set_title("Nelder–Mead: Simplex evolution (static)")

    # draw sampled simplexes
    for k in range(0, len(history), max(1, every)):
        plot_simplex(ax, history[k], linewidth=0.8, alpha=0.6)

    # last simplex emphasized
    plot_simplex(ax, history[-1], linewidth=2.0, alpha=1.0)

    if show_best_path:
        # best vertex per iteration = argmin f among 3 vertices
        best_points = []
        for verts in history:
            vals = np.array([f(verts[i]) for i in range(3)], dtype=float)
            best_points.append(verts[np.argmin(vals)])
        best_points = np.array(best_points)
        ax.plot(best_points[:, 0], best_points[:, 1], linewidth=2.0)

    ax.grid(True, alpha=0.2)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"Saved static plot to: {savepath}")

    plt.show()


def animate_history(
    f,
    history: list[Array],
    xlim=(-6, 6),
    ylim=(-6, 6),
    ngrid: int = 250,
    levels: int = 40,
    interval_ms: int = 60,
    stride: int = 1,
    savepath: str | None = None,
):
    """
    Animation:
      - draws contours once
      - updates simplex triangle per frame
      - stride lets you skip iterations (e.g., stride=5)
    Save:
      - GIF: savepath="out.gif" (needs pillow installed)
      - MP4: savepath="out.mp4" (needs ffmpeg installed)
    """
    # precompute contours once
    fig, ax = plt.subplots()
    plot_contours(ax, f, xlim, ylim, ngrid=ngrid, levels=levels)
    ax.set_title("Nelder–Mead: Simplex evolution (animation)")
    ax.grid(True, alpha=0.2)

    # prepare line artist for triangle
    line, = ax.plot([], [], linewidth=2.0, marker="o")

    # frames are indices into history
    frame_indices = list(range(0, len(history), max(1, stride)))

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame_i: int):
        verts = history[frame_indices[frame_i]]
        tri = _close_triangle(verts)
        line.set_data(tri[:, 0], tri[:, 1])
        ax.set_title(f"Nelder–Mead: iter {frame_indices[frame_i]}/{len(history)-1}")
        return (line,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    if savepath:
        if savepath.lower().endswith(".gif"):
            anim.save(savepath, writer="pillow", dpi=150)
        elif savepath.lower().endswith(".mp4"):
            anim.save(savepath, writer="ffmpeg", dpi=150)
        else:
            raise ValueError("savepath must end with .gif or .mp4")
        print(f"Saved animation to: {savepath}")

    plt.show()


# ----------------------------
# Beispiel-Nutzung
# ----------------------------
if __name__ == "__main__":
    # Importiere deine Implementierung
    from nelder_mead import simplex, himmelblau

    # Run optimizer with history
    x_min, f_min, N, hist = simplex(himmelblau, [0, 0], N_max=2000, p=1e-12, return_history=True)
    print("x_min:", x_min, "f_min:", f_min, "N:", N, "iters_saved:", len(hist))

    # 1) Static plot (PNG)
    #plot_history_static(
     #   himmelblau,
     #   hist,
     #   xlim=(-6, 6),
     #   ylim=(-6, 6),
     #   every=10,
      #  savepath="simplex_himmelblau.png",
    #)

    # 2) Animation (GIF) – optional
    animate_history(
         himmelblau,
         hist,
         xlim=(-6, 6),
         ylim=(-6, 6),
        stride=3,
         interval_ms=180,
         savepath="simplex_himmelblau.gif",
     )
