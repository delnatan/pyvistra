"""Figures for docs/imaris-surfaces.md.

Everything is synthetic: a smooth two-lobed shape is encoded with
``SurfaceOctree.from_signed_distance`` and drawn back through pyvistra's
decoder, so the pictures show exactly what the reader does.

    python docs/imaris-surfaces/make_figures.py   # needs matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pyvistra.readers.imaris_surfaces import (
    SurfaceOctree,
    decode_distance,
    decode_normal,
)

OUT = Path(__file__).parent

# reference palette (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
BLUE_LIGHT = "#cde2fb"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
AQUA_LIGHT = "#c6eddd"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
    }
)


def peanut_octree():
    """Two smoothly joined spheres on a 16 x 24 x 32 cell lattice of 1 um."""
    shape = (16, 24, 32)
    zz, yy, xx = np.meshgrid(*[np.arange(n + 1.0) for n in shape], indexing="ij")
    p = np.stack([zz, yy, xx], -1)
    d1 = 6.5 - np.linalg.norm(p - (8.0, 12.0, 10.5), axis=-1)
    d2 = 5.0 - np.linalg.norm(p - (8.0, 11.0, 21.0), axis=-1)
    k = 4.0  # smooth maximum: union of the two lobes
    h = np.clip(0.5 + 0.5 * (d1 - d2) / k, 0, 1)
    sdf = d2 + (d1 - d2) * h + k * h * (1 - h)
    return SurfaceOctree.from_signed_distance(sdf, (0, 0, 0), (1, 1, 1)), sdf


def fig_octree(octree, sdf):
    """Cell states on the z = 8 slab and the closest-point vectors."""
    z = 8
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.6), constrained_layout=True)
    for level in octree.levels:
        s = level.cell_size
        for kind, cells in (("full", level.full), ("crossed", level.crossed)):
            for o in cells:
                if not (o[0] <= z < o[0] + s[0]):
                    continue
                finest = s[2] == 1
                if kind == "full":
                    face, edge = AQUA_LIGHT, AQUA
                elif finest:
                    face, edge = BLUE_LIGHT, BLUE
                else:
                    face, edge = "none", MUTED
                ax.add_patch(Rectangle((o[2], o[1]), s[2], s[1], facecolor=face,
                                       edgecolor=edge, linewidth=0.6))
    yy, xx = np.mgrid[0 : 24 : 0.05, 0 : 32 : 0.05]
    pts = np.stack([np.full(xx.size, z + 0.5), yy.ravel(), xx.ravel()], 1)
    ax.contour(xx, yy, octree.sample(pts).reshape(xx.shape), [0], colors=INK, linewidths=1.2)
    ax.set(xlim=(0, 32), ylim=(24, 0), aspect="equal", xlabel="x (cells)", ylabel="y (cells)",
           title="Octree cells crossing the slab z = 8")
    handles = [Rectangle((0, 0), 1, 1, fc=AQUA_LIGHT, ec=AQUA), Rectangle((0, 0), 1, 1, fc=BLUE_LIGHT, ec=BLUE),
               Rectangle((0, 0), 1, 1, fc="none", ec=MUTED), plt.Line2D([], [], color=INK, lw=1.2)]
    ax.legend(handles, ["fully inside (any level)", "crossed, finest level", "crossed, refined", "zero set"],
              loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=8)

    # closest-point vectors on lattice plane z = 8, zoomed on the waist
    fine = octree.levels[-1]
    radius = np.linalg.norm(octree.spacing)  # from_signed_distance scales distances by the cell diagonal
    seen = {}
    for o, codes in zip(fine.crossed, fine.codes):
        for c in range(8):
            v = tuple(o + [(c >> 2) & 1, (c >> 1) & 1, c & 1])
            if v[0] == z:
                seen[v] = (decode_distance(codes[c]) * radius, decode_normal(codes[c]))
    x0, x1, y0, y1 = 11, 22, 5, 18
    bx.contour(xx, yy, octree.sample(np.stack([np.full(xx.size, z), yy.ravel(), xx.ravel()], 1)).reshape(xx.shape),
               [0], colors=INK, linewidths=1.2)
    for (vz, vy, vx), (s, n) in seen.items():
        if not (x0 <= vx <= x1 and y0 <= vy <= y1):
            continue
        tip = np.array([vx, vy]) + s * n[[2, 1]]
        inside = s > 0
        bx.add_patch(FancyArrowPatch((vx, vy), tip, arrowstyle="-|>", mutation_scale=7, color=ORANGE, lw=1.0))
        bx.plot(vx, vy, "o", ms=4.5, mfc=BLUE if inside else SURFACE, mec=BLUE, mew=1.0)
    for g in range(x0, x1 + 1):
        bx.axvline(g, color=GRID, lw=0.5, zorder=0)
    for g in range(y0, y1 + 1):
        bx.axhline(g, color=GRID, lw=0.5, zorder=0)
    bx.set(xlim=(x0 - 0.5, x1 + 0.5), ylim=(y1 + 0.5, y0 - 0.5), aspect="equal", xlabel="x (cells)",
           title="Vector to the closest surface point, per corner")
    bx.legend([plt.Line2D([], [], ls="", marker="o", mfc=BLUE, mec=BLUE),
               plt.Line2D([], [], ls="", marker="o", mfc=SURFACE, mec=BLUE),
               plt.Line2D([], [], color=ORANGE), plt.Line2D([], [], color=INK, lw=1.2)],
              ["corner inside", "corner outside", "s · n (projected)", "zero set"],
              loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=8)
    fig.savefig(OUT / "octree.png", dpi=160)
    plt.close(fig)


def _bit_row(ax, y, fields, total, title):
    """fields: (first_bit, n_bits, label, fill) drawn LSB on the right."""
    for first, n, label, fill in fields:
        x = total - first - n
        ax.add_patch(Rectangle((x, y), n, 1, facecolor=fill, edgecolor=SURFACE, linewidth=2))
        ax.text(x + n / 2, y + 0.5, label, ha="center", va="center", fontsize=8.5, color=INK)
        ax.text(x + n - 0.15, y - 0.12, str(first), ha="right", va="top", fontsize=7, color=MUTED)
        ax.text(x + 0.15, y - 0.12, str(first + n - 1), ha="left", va="top", fontsize=7, color=MUTED)
    ax.text(0, y + 1.25, title, ha="left", va="bottom", fontsize=9.5, fontweight="semibold", color=INK)


def fig_bits():
    fig, ax = plt.subplots(figsize=(9.6, 3.9), constrained_layout=True)
    _bit_row(ax, 4.6, [
        (56, 8, "data mask", BLUE_LIGHT), (32, 24, "first data value (÷ 8 = record index)", "#e8f0fb"),
        (24, 8, "child mask", AQUA_LIGHT), (0, 24, "first child node (in the child's block)", "#e5f5ee"),
    ], 64, "Node word (uint64, BlockData[NodeBegin:NodeEnd])")
    ax.text(64, 6.85, "bit 63 … 0", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    # corner code drawn on a 64-wide canvas at 2x scale
    scale = 2
    for first, n, label, fill in [(24, 8, "tangent 2 (byte 3)", "#fde5da"), (16, 8, "tangent 1 (byte 2)", "#fde5da"),
                                  (13, 3, "face", "#fbcbb8"), (0, 13, "√ distance", BLUE_LIGHT)]:
        x = (32 - first - n) * scale
        ax.add_patch(Rectangle((x, 1.4), n * scale, 1, facecolor=fill, edgecolor=SURFACE, linewidth=2))
        ax.text(x + n * scale / 2, 1.9, label, ha="center", va="center", fontsize=8.5, color=INK)
        ax.text(x + n * scale - 0.3, 1.28, str(first), ha="right", va="top", fontsize=7, color=MUTED)
        ax.text(x + 0.3, 1.28, str(first + n - 1), ha="left", va="top", fontsize=7, color=MUTED)
    ax.text(0, 2.65, "VCP corner code (uint32; one per cell corner)", ha="left", va="bottom",
            fontsize=9.5, fontweight="semibold", color=INK)
    ax.text(0, 0.35, "Stored per record as 8 × uint16 (bits 0-15), then 8 × (byte 2, byte 3) "
            "→ 32 bytes per cell. A record of all 0xFF bytes marks a fully-inside cell.",
            ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set(xlim=(-0.5, 64.5), ylim=(0, 7.2))
    ax.axis("off")
    fig.savefig(OUT / "bits.png", dpi=160)
    plt.close(fig)


def fig_encodings():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    code = np.arange(8192)
    s = decode_distance(code)
    ax.plot(code, s, color=BLUE, lw=2)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(4095.5, color=GRID, lw=0.8, zorder=0)
    ticks = np.arange(0, 8192, 512)
    ax.plot(ticks, decode_distance(ticks), "o", ms=3.5, color=BLUE)
    ax.set(xlabel="13-bit distance field", ylabel="decoded value  sign(x)·x²",
           title="Distance: square-root code, fine near the surface")
    ax.text(4300, -0.55, "codes every 512 bunch up\nnear the zero crossing", fontsize=8, color=INK2)
    ax.spines[["top", "right"]].set_visible(False)

    # cube-map normal: 2D cut through the x-y plane
    bx.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor=MUTED, lw=1))
    t = np.linspace(0, 2 * np.pi, 200)
    bx.plot(np.cos(t), np.sin(t), color=GRID, lw=1)
    for k, (dx, dy) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
        lbl = ["face 1 (+x)", "face 2 (+y)", "face 4 (−x)", "face 5 (−y)"][k]
        bx.text(1.18 * dx, 1.18 * dy, lbl, ha="center", va="center", fontsize=8, color=INK2,
                rotation=90 if dx else 0)
    for d in (1, -1):
        bx.plot([-1.0, 1.0], [d * 1.0, -d * 1.0], color=GRID, lw=0.8, ls="--")
    n = np.array([np.cos(0.5), np.sin(0.5)])
    tan = n[1] / n[0]
    bx.add_patch(FancyArrowPatch((0, 0), tuple(n), arrowstyle="-|>", mutation_scale=10, color=ORANGE, lw=1.6))
    bx.plot([1, 1], [0, tan], color=BLUE, lw=2.2)
    bx.plot(1, tan, "o", ms=5, color=BLUE)
    bx.text(0.42, 0.36, "n", color=ORANGE, fontsize=10, fontstyle="italic", ha="center", va="bottom")
    bx.text(0.94, -0.3, f"t = n_y / |n_x| = {tan:.2f}\nbyte = (t + 1) · 127.5", ha="right", va="center",
            fontsize=8, color=INK)
    bx.annotate("", xy=(1.0, tan / 2), xytext=(0.8, -0.15),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    bx.set(xlim=(-1.45, 1.45), ylim=(-1.35, 1.35), aspect="equal",
           title="Normal: major-axis face + two tangents")
    bx.axis("off")
    fig.savefig(OUT / "encodings.png", dpi=160)
    plt.close(fig)


def fig_mesh(octree):
    mesh = octree.mesh()
    light = np.array([0.55, -0.45, 0.70])
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.clip(mesh.normals[mesh.faces].mean(1) @ light, 0, 1)
    base = np.array([0x2A, 0x78, 0xD6]) / 255
    colors = np.clip(base[None] * shade[:, None] + (1 - shade[:, None]) * 0.12, 0, 1)
    fig = plt.figure(figsize=(9.6, 4.4), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    xyz = mesh.vertices[:, ::-1]
    ax.add_collection3d(Poly3DCollection(xyz[mesh.faces], facecolors=colors, edgecolors="none"))
    ax.set(xlim=(4, 27), ylim=(3, 21), zlim=(1, 15))
    ax.set_box_aspect((23, 18, 14), zoom=1.35)
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    ax.set_title(f"Marching cubes on the lattice: {len(mesh.faces):,} triangles,\nshaded with the stored VCP normals",
                 fontsize=9.5)
    bx = fig.add_subplot(1, 2, 2)
    lat = octree.lattice[8]
    im = bx.imshow(lat, cmap="Blues", vmin=-1, vmax=1, origin="upper")
    bx.contour(lat, [0], colors=INK, linewidths=1.2)
    bx.set(title="Vertex lattice at z = 8 (+1 inside, −1 outside,\ndecoded distance on crossed-cell corners)",
           xlabel="x (vertex)", ylabel="y (vertex)")
    cb = fig.colorbar(im, ax=bx, shrink=0.8)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "mesh.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    octree, sdf = peanut_octree()
    fig_bits()
    fig_encodings()
    fig_octree(octree, sdf)
    fig_mesh(octree)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
