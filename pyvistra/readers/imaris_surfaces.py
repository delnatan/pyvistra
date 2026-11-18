"""Imaris Surfaces ("MegaSurfaces", Imaris >= 9) decoding.

Imaris stores every surface object as a sparse octree of Vector-to-Closest-
Point (VCP) samples (Demir & Westermann, VMV 2015). The surface is the
marching-cubes zero set of a scalar lattice: corners of cells crossed by the
surface carry an encoded signed distance, fully-inside cells count as
positive, absent cells as negative. Decoding that lattice reproduces Imaris'
own triangle count, area and volume exactly. The byte-level specification is
in ``docs/imaris-surfaces.md``.

Conventions: every array and coordinate here is in (z, y, x) order. Octree
corner ``c`` of a cell sits at offset ``CORNER_OFFSETS[c]`` -- in the file,
bit 0 of ``c`` is x, bit 1 is y and bit 2 is z.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

# (z, y, x) offset of octree child / cell corner c (file order: x is bit 0).
CORNER_OFFSETS = np.array([[c >> 2 & 1, c >> 1 & 1, c & 1] for c in range(8)])

FULL_CODE = 0xFFFFFFFF  # every corner of a fully-inside cell's record
_DIST_BITS = 0x1FFF
_DIST_MID = 4095.5  # (2**13 - 1) / 2: code of distance zero
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


# --------------------------------------------------------------------------
# VCP corner codes
# --------------------------------------------------------------------------


def decode_distance(codes):
    """Encoded signed distance of VCP corner codes, positive inside.

    The 13-bit field stores ``sqrt`` of the normalized distance (as in Demir &
    Westermann), so the value returned is ``sign(x) * x**2`` with
    ``x = (field - 4095.5) / 4095.5`` in [-1, 1]. Its physical scale is not
    stored in the file and is not needed: the surface is the zero set.
    """
    x = ((np.asarray(codes) & _DIST_BITS) - _DIST_MID) / _DIST_MID
    return np.sign(x) * x * x


def decode_normal(codes):
    """Outward unit normals (..., 3) in (z, y, x) of VCP corner codes.

    Bits 13-15 hold the cube face of the normal's major axis; bytes 2 and 3
    hold the two minor components divided by the major one, mapped from
    [-1, 1] to [0, 255]. In (x, y, z) terms the minor axes of major axis
    ``a`` are ``(a + 1) % 3`` and ``(a + 2) % 3``.
    """
    codes = np.asarray(codes, dtype=np.uint32)
    face = (codes & 0xFFFF) >> 13
    t1 = ((codes >> 16) & 0xFF) / 127.5 - 1.0
    t2 = (codes >> 24) / 127.5 - 1.0
    a_xyz = (face.astype(np.int64) - 1) % 3  # 0=x, 1=y, 2=z
    n_xyz = np.zeros(codes.shape + (3,))
    np.put_along_axis(
        n_xyz, a_xyz[..., None], np.where(face <= 3, 1.0, -1.0)[..., None], -1
    )
    np.put_along_axis(n_xyz, ((a_xyz + 1) % 3)[..., None], t1[..., None], -1)
    np.put_along_axis(n_xyz, ((a_xyz + 2) % 3)[..., None], t2[..., None], -1)
    n_xyz /= np.linalg.norm(n_xyz, axis=-1, keepdims=True)
    return n_xyz[..., ::-1]


def encode_corners(distance, normal):
    """Inverse of :func:`decode_distance` / :func:`decode_normal`.

    distance: (...,) signed distances in [-1, 1] (positive inside), already
        normalized by whatever radius the caller chose.
    normal: (..., 3) outward normals in (z, y, x); need not be unit length.
    """
    d = np.clip(np.asarray(distance, float), -1.0, 1.0)
    x = np.sign(d) * np.sqrt(np.abs(d))
    dist_field = np.clip(
        np.rint(x * _DIST_MID + _DIST_MID), 0, _DIST_BITS
    ).astype(np.uint32)
    n_xyz = np.asarray(normal, float)[..., ::-1]
    a = np.argmax(np.abs(n_xyz), axis=-1)

    def component(axis):
        return np.take_along_axis(n_xyz, axis[..., None], -1)[..., 0]

    major = component(a)
    face = np.where(major >= 0, a + 1, a + 4).astype(np.uint32)
    scale = np.where(major == 0, 1.0, np.abs(major))

    def byte(t):
        return np.clip(np.rint((t + 1.0) * 127.5), 0, 255).astype(np.uint32)

    b1 = byte(component((a + 1) % 3) / scale)
    b2 = byte(component((a + 2) % 3) / scale)
    return (face << 13) | dist_field | (b1 << 16) | (b2 << 24)


# --------------------------------------------------------------------------
# Image grid and meshes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageGrid:
    """Voxel grid of an image: voxel ``i`` spans ``origin + [i, i+1) * voxel_size``."""

    origin: np.ndarray  # (z, y, x) um, corner of voxel 0 (Imaris ExtMin)
    voxel_size: np.ndarray  # (z, y, x) um
    shape: tuple[int, int, int]  # (z, y, x)

    def to_voxel(self, points):
        """Physical points (N, 3) -> voxel index coordinates (voxel centers at integers)."""
        return (
            np.asarray(points, float) - self.origin
        ) / self.voxel_size - 0.5

    def to_physical(self, coords):
        return (
            np.asarray(coords, float) + 0.5
        ) * self.voxel_size + self.origin


@dataclass(frozen=True)
class SurfaceMesh:
    """Triangle mesh with outward-facing winding and outward unit normals."""

    vertices: np.ndarray  # (V, 3) float, (z, y, x)
    faces: np.ndarray  # (F, 3) int
    normals: np.ndarray  # (V, 3) float, (z, y, x)

    @property
    def area(self):
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()

    @property
    def volume(self):
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        # (z, y, x) is a left-handed reading of a right-handed frame: negate.
        return -np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


# --------------------------------------------------------------------------
# One surface object at one timepoint
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OctreeLevel:
    """Cells of one octree level, in finest-cell units (z, y, x)."""

    cell_size: np.ndarray  # (3,) int
    full: np.ndarray  # (F, 3) int origins of fully-inside leaf cells
    crossed: np.ndarray  # (C, 3) int origins of cells crossed by the surface
    codes: np.ndarray  # (C, 8) uint32 VCP corner codes of the crossed cells


@dataclass(frozen=True, eq=False)
class SurfaceOctree:
    """One surface object at one timepoint, decoded from its VCP octree.

    Geometry lives on the finest level: ``levels[-1].crossed`` cells hold the
    surface, ``full`` cells of every level are inside, everything else is
    outside. Crossed cells of coarser levels are Imaris' level-of-detail
    records; their normals are valid but many of their distances are zero
    placeholders, so they don't define a surface.
    """

    object_id: int
    time_index: int
    origin: np.ndarray  # (z, y, x) um, physical position of lattice vertex 0
    spacing: np.ndarray  # (z, y, x) um per finest cell
    shape: tuple[int, int, int]  # finest cells (z, y, x)
    levels: tuple[OctreeLevel, ...]  # root first

    @property
    def extent(self):
        """Physical (z, y, x) position of the far lattice corner."""
        return self.origin + np.array(self.shape) * self.spacing

    @cached_property
    def lattice(self):
        """Scalar vertex lattice, shape ``self.shape + 1``; the surface is its zero set.

        Corners of crossed finest cells hold :func:`decode_distance` values,
        corners of fully-inside cells +1, everything else -1.
        """
        lat = np.full(tuple(np.add(self.shape, 1)), -1.0, dtype=np.float32)
        for level in self.levels:
            _fill_boxes(lat, level.full, level.cell_size + 1, 1.0)
        fine = self.levels[-1]
        corners = fine.crossed[:, None, :] + CORNER_OFFSETS[None]
        lat[corners[..., 0], corners[..., 1], corners[..., 2]] = (
            decode_distance(fine.codes)
        )
        return lat

    def sample(self, points):
        """Interpolated lattice value at physical points (N, 3); > 0 is inside."""
        from scipy.ndimage import map_coordinates

        u = (np.asarray(points, float) - self.origin) / self.spacing
        return map_coordinates(
            self.lattice, u.T, order=1, mode="constant", cval=-1.0
        )

    def rasterize(self, grid: ImageGrid, coverage: float | None = None):
        """Boolean mask of this object on an image grid.

        A voxel is inside when its center is inside the surface, or, with
        ``coverage`` set, when at least that fraction of its volume is
        (estimated from 4x4x4 sub-samples, only for voxels near the surface).
        ``coverage=0.25`` reproduces Imaris' own surface-to-mask-channel
        export (Dice ~0.99).

        Returns ``(mask, slices)``: the mask covers ``image[slices]``.
        """
        pad = 0 if coverage is None else 1
        lo = (
            np.ceil(
                (self.origin - grid.origin) / grid.voxel_size - 0.5
            ).astype(int)
            - pad
        )
        hi = (
            np.floor(
                (self.extent - grid.origin) / grid.voxel_size - 0.5
            ).astype(int)
            + 1
            + pad
        )
        lo = np.clip(lo, 0, grid.shape)
        hi = np.clip(hi, lo, grid.shape)
        slices = tuple(slice(a, b) for a, b in zip(lo, hi))
        axes = [
            grid.origin[i]
            + (np.arange(lo[i], hi[i]) + 0.5) * grid.voxel_size[i]
            for i in range(3)
        ]
        mask = _grid_values(self, axes) > 0
        if coverage is not None and mask.size:
            corner_axes = [a - 0.5 * v for a, v in zip(axes, grid.voxel_size)]
            corner_axes = [
                np.append(a, a[-1] + v)
                for a, v in zip(corner_axes, grid.voxel_size)
            ]
            corners = _grid_values(self, corner_axes) > 0
            # voxels whose corners or center disagree straddle the surface
            mixed = np.zeros(mask.shape, bool)
            nz, ny, nx = mask.shape
            for dz, dy, dx in CORNER_OFFSETS:
                mixed |= (
                    corners[dz : dz + nz, dy : dy + ny, dx : dx + nx] != mask
                )
            idx = np.argwhere(mixed)
            if len(idx):
                k = 4
                sub = (np.arange(k) + 0.5) / k - 0.5
                offsets = np.stack(
                    np.meshgrid(sub, sub, sub, indexing="ij"), -1
                ).reshape(-1, 3)
                centers = np.stack([axes[i][idx[:, i]] for i in range(3)], 1)
                inside = np.zeros(len(idx))
                for o in offsets:
                    inside += self.sample(centers + o * grid.voxel_size) > 0
                mask[tuple(idx.T)] = inside / len(offsets) >= coverage
        return mask, slices

    def mesh(self):
        """Marching-cubes mesh of the zero set, in physical (z, y, x) um.

        Normals come from the stored VCP normals, interpolated along each
        mesh vertex's lattice edge. The triangle count and area equal Imaris'
        ``NumberOfTriangles`` and ``Area`` statistics.
        """
        from skimage.measure import marching_cubes

        padded = np.pad(self.lattice, 1, constant_values=-1.0)
        verts, faces, grad_normals, _ = marching_cubes(padded, 0.0)
        verts -= 1.0  # undo padding; now in lattice index units
        # skimage's gradient normals point down the field, i.e. outward here,
        # and its winding agrees with them.
        normals = self._vertex_normals(verts, grad_normals)
        return SurfaceMesh(
            vertices=self.origin + verts * self.spacing,
            faces=np.ascontiguousarray(faces),
            normals=normals,
        )

    def _vertex_normals(self, verts, fallback):
        fine = self.levels[-1]
        if len(fine.crossed) == 0:
            return fallback
        dims = np.add(self.shape, 1)
        corner_idx = fine.crossed[:, None, :] + CORNER_OFFSETS[None]
        keys = np.ravel_multi_index(tuple(corner_idx.reshape(-1, 3).T), dims)
        keys, first = np.unique(keys, return_index=True)
        corner_normals = decode_normal(fine.codes).reshape(-1, 3)[first]

        # Each marching-cubes vertex lies on one lattice edge, between the
        # floor and ceil of its single fractional coordinate.
        a = np.floor(verts + 1e-6).astype(int)
        frac = verts - a
        axis = np.argmax(frac, axis=1)
        t = frac[np.arange(len(verts)), axis]
        b = a.copy()
        b[np.arange(len(verts)), axis] += t > 1e-6
        a = np.clip(a, 0, dims - 1)
        b = np.clip(b, 0, dims - 1)

        def lookup(p):
            k = np.ravel_multi_index(tuple(p.T), dims)
            i = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
            return np.where((keys[i] == k)[:, None], corner_normals[i], np.nan)

        n = (1 - t)[:, None] * lookup(a) + t[:, None] * lookup(b)
        n = np.where(np.isnan(n), fallback, n)
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    @classmethod
    def from_signed_distance(
        cls, sdf, origin, spacing, *, object_id=0, time_index=0, normals=None
    ):
        """Build an octree from a signed-distance lattice (positive inside, um).

        sdf: (nz+1, ny+1, nx+1) values at the lattice vertices of an
            (nz, ny, nx)-cell grid. normals: optional (..., 3) outward normals;
            defaults to the negated sdf gradient. Distances are stored relative
            to the cell diagonal, so values beyond one diagonal saturate.
        """
        sdf = np.asarray(sdf, float)
        spacing = np.asarray(spacing, float)
        if normals is None:
            normals = -np.stack(np.gradient(sdf, *spacing), axis=-1)
        shape = tuple(int(n) - 1 for n in sdf.shape)
        inside = sdf > 0
        corner_inside = np.stack(
            [
                inside[
                    dz : dz + shape[0], dy : dy + shape[1], dx : dx + shape[2]
                ]
                for dz, dy, dx in CORNER_OFFSETS
            ]
        )
        full = corner_inside.all(0)
        crossed = corner_inside.any(0) & ~full

        # Isotropic 2x2x2 splits; level k holds the children of level k-1's
        # crossed cells (level 0: children of the root, which has no record).
        n_levels = max(int(np.ceil(np.log2(max(max(shape), 2)))), 1)
        dims = 2**n_levels
        state = np.zeros((dims,) * 3, np.int8)  # 0 empty, 1 full, 2 crossed
        state[: shape[0], : shape[1], : shape[2]][full] = 1
        state[: shape[0], : shape[1], : shape[2]][crossed] = 2
        radius = np.linalg.norm(spacing)
        levels = []
        prev_crossed = np.ones((1, 1, 1), bool)  # the root
        for lev in range(1, n_levels + 1):
            size = dims >> lev
            coarse = state.reshape(
                dims // size, size, dims // size, size, dims // size, size
            )
            is_full = (coarse == 1).all(axis=(1, 3, 5))
            is_empty = (coarse == 0).all(axis=(1, 3, 5))
            parent_crossed = np.repeat(
                np.repeat(np.repeat(prev_crossed, 2, 0), 2, 1), 2, 2
            )
            is_crossed = ~is_full & ~is_empty & parent_crossed
            # a full cell is a leaf only where its parent was crossed
            leaf_full = is_full & parent_crossed
            org = np.argwhere(is_crossed) * size
            crn = org[:, None, :] + CORNER_OFFSETS[None] * size
            crn = np.minimum(crn, np.array(shape))
            d = sdf[crn[..., 0], crn[..., 1], crn[..., 2]] / (radius * size)
            nrm = normals[crn[..., 0], crn[..., 1], crn[..., 2]]
            levels.append(
                OctreeLevel(
                    cell_size=np.full(3, size),
                    full=np.argwhere(leaf_full) * size,
                    crossed=org,
                    codes=encode_corners(d, nrm),
                )
            )
            prev_crossed = is_crossed
        return cls(
            object_id=object_id,
            time_index=time_index,
            origin=np.asarray(origin, float),
            spacing=spacing,
            shape=shape,
            levels=tuple(levels),
        )


def _fill_boxes(lat, origins, size, value):
    """lat[o : o + size] = value for every origin o (boxes of one common size)."""
    if len(origins) == 0:
        return
    if np.prod(size) <= 64:
        offs = np.argwhere(np.ones(size, bool))
        pts = (origins[:, None, :] + offs[None]).reshape(-1, 3)
        lat[pts[:, 0], pts[:, 1], pts[:, 2]] = value
    else:
        for z, y, x in origins:
            lat[z : z + size[0], y : y + size[1], x : x + size[2]] = value


def _grid_values(octree, axes):
    """Lattice values on the tensor grid of physical (z, y, x) axis coordinates."""
    from scipy.ndimage import map_coordinates

    shape = tuple(len(a) for a in axes)
    if 0 in shape:
        return np.zeros(shape)
    u = [(a - o) / s for a, o, s in zip(axes, octree.origin, octree.spacing)]
    coords = np.meshgrid(*u, indexing="ij")
    return map_coordinates(
        octree.lattice, coords, order=1, mode="constant", cval=-1.0
    )


# --------------------------------------------------------------------------
# File decoding
# --------------------------------------------------------------------------


def _decode_model(group, model):
    """Decode one SurfaceModel row of a MegaSurfaces group into a SurfaceOctree."""
    infos = group["BlockInfo"][model["BlockInfoBegin"] : model["BlockInfoEnd"]]
    li = group["LevelInfo"][model["LevelInfoBegin"] : model["LevelInfoEnd"]]
    split = np.stack(
        [li["SamplingZ"], li["SamplingY"], li["SamplingX"]], axis=1
    ).astype(int)
    shape = (int(model["SizeZ"]), int(model["SizeY"]), int(model["SizeX"]))
    emin = np.array(
        [model["ExtendMinZ"], model["ExtendMinY"], model["ExtendMinX"]], float
    )
    emax = np.array(
        [model["ExtendMaxZ"], model["ExtendMaxY"], model["ExtendMaxX"]], float
    )
    spacing = (emax - emin) / np.array(shape)

    # Blocks form a tree: a level that grew too large is split into several
    # blocks, each holding the children of a contiguous range of parent nodes
    # (cumulative end indices in SplitOffset). Child blocks are listed in
    # BlockInfo in the same order as those ranges.
    split_offsets = group["SplitOffset"]
    children = {}
    for j in range(1, len(infos)):
        children.setdefault(int(infos[j]["ParentBlockIndex"]), []).append(j)
    bounds = {
        b: split_offsets[
            infos[b]["SplitOffsetsBegin"] : infos[b]["SplitOffsetsEnd"]
        ]["Data"]
        for b in children
    }

    words = _ModelWords(group["BlockData"], infos)
    cell = np.prod(split, axis=0)
    blk = np.zeros(1, int)
    idx = np.zeros(1, int)
    org = np.zeros((1, 3), int)
    levels = []
    for lev in range(len(split)):
        cell = cell // split[lev]
        full, crossed, codes = [], [], []
        nxt_blk, nxt_idx, nxt_org = [], [], []
        for b in np.unique(blk):
            sel = blk == b
            node = words.nodes(b)[idx[sel]]
            lo, hi = node & 0xFFFFFFFF, node >> 32
            child_mask = (lo >> 24).astype(np.int64)
            child_off = (lo & 0xFFFFFF).astype(np.int64)
            data_mask = (hi >> 24).astype(np.int64)
            data_rec = (hi & 0xFFFFFF).astype(
                np.int64
            ) // 8  # offsets count corner values
            leaves = words.leaves(b)
            if b in children:
                child_block = np.array(children[b])[
                    np.searchsorted(bounds[b], idx[sel], side="right")
                ]
            for c in range(8):
                below = (1 << c) - 1
                o = org[sel] + CORNER_OFFSETS[c] * cell
                has_child = (child_mask >> c) & 1 == 1
                has_data = (data_mask >> c) & 1 == 1
                if has_child.any():
                    nxt_blk.append(child_block[has_child])
                    nxt_idx.append(
                        child_off[has_child]
                        + _POPCOUNT[child_mask[has_child] & below]
                    )
                    nxt_org.append(o[has_child])
                if has_data.any():
                    rec = leaves[
                        data_rec[has_data]
                        + _POPCOUNT[data_mask[has_data] & below]
                    ]
                    code = (
                        rec[:, :16].view("<u2").astype(np.uint32)
                        | (rec[:, 16::2].astype(np.uint32) << 16)
                        | (rec[:, 17::2].astype(np.uint32) << 24)
                    )
                    is_full = (code == FULL_CODE).all(axis=1)
                    full.append(o[has_data][is_full & ~has_child[has_data]])
                    crossed.append(o[has_data][~is_full])
                    codes.append(code[~is_full])
        levels.append(
            OctreeLevel(
                cell_size=cell.copy(),
                full=np.concatenate(full) if full else np.zeros((0, 3), int),
                crossed=np.concatenate(crossed)
                if crossed
                else np.zeros((0, 3), int),
                codes=np.concatenate(codes)
                if codes
                else np.zeros((0, 8), np.uint32),
            )
        )
        if not nxt_blk:
            break
        blk, idx, org = (
            np.concatenate(nxt_blk),
            np.concatenate(nxt_idx),
            np.concatenate(nxt_org),
        )

    # Cells beyond SizeX/Y/Z can't hold data; drop them so every cell is in range.
    def clip(level):
        keep_c = np.all(level.crossed < shape, axis=1)
        return OctreeLevel(
            level.cell_size,
            level.full[np.all(level.full < shape, axis=1)],
            level.crossed[keep_c],
            level.codes[keep_c],
        )

    return SurfaceOctree(
        object_id=int(model["ID"]),
        time_index=int(model["ID_Time"]),
        origin=emin,
        spacing=spacing,
        shape=shape,
        levels=tuple(clip(level) for level in levels),
    )


class _ModelWords:
    """Reads only the BlockData rows one model's blocks live in."""

    def __init__(self, dataset, infos):
        row = dataset.shape[1]
        start = int(min(infos["NodeBegin"].min(), infos["LeafBegin"].min()))
        end = int(max(infos["NodeEnd"].max(), infos["LeafEnd"].max()))
        r0, r1 = start // row, max(end - 1, start) // row + 1
        self._base = r0 * row
        self._flat = dataset[r0:r1].ravel().view(np.uint64)
        self._infos = infos

    def nodes(self, b):
        i = self._infos[b]
        return self._flat[
            i["NodeBegin"] - self._base : i["NodeEnd"] - self._base
        ]

    def leaves(self, b):
        i = self._infos[b]
        return (
            self._flat[i["LeafBegin"] - self._base : i["LeafEnd"] - self._base]
            .view(np.uint8)
            .reshape(-1, 32)
        )


def _attr_text(attrs, key, default=""):
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, np.ndarray):
        v = (
            b"".join(v.ravel().tolist())
            if v.dtype.kind == "S"
            else v.ravel()[0]
        )
    return (
        v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
    )


class ImarisSurfaces:
    """One Surfaces item of an Imaris scene: every object at every timepoint.

    Objects are rows of the item's SurfaceModel table, addressed by index
    (0..len-1). ``object_ids`` and ``time_indices`` describe each row;
    ``statistics`` holds Imaris' per-object values (Volume, Area, ...).
    Masks, labels and meshes are on the image grid of the reader the item
    came from.
    """

    def __init__(self, group, grid_for_level):
        self._group = group
        self._grid = grid_for_level
        self.name = _attr_text(
            group.attrs, "Name", group.name.rsplit("/", 1)[-1]
        )

    def __repr__(self):
        times = np.unique(self.time_indices)
        span = f"t={times.min()}..{times.max()}" if len(times) else "empty"
        return f"<ImarisSurfaces {self.name!r}: {len(self)} objects, {span}>"

    def __len__(self):
        return len(self._models)

    @cached_property
    def _models(self):
        return self._group["SurfaceModel"][()]

    @property
    def object_ids(self):
        return self._models["ID"]

    @property
    def time_indices(self):
        return self._models["ID_Time"]

    @cached_property
    def statistics(self):
        """Per-object statistics as ``{name: array}`` aligned with the object index."""
        if "SurfaceModelInfo" not in self._group:
            return {}
        info = self._group["SurfaceModelInfo"][()]
        return {k: info[k] for k in info.dtype.names}

    @cached_property
    def tracks(self):
        """``{track_id: object_ids}`` for tracked surfaces (empty otherwise)."""
        if "Track0" not in self._group:
            return {}
        tracks = self._group["Track0"][()]
        objects = self._group["TrackObject0"][()]["ID_Object"]
        return {
            int(t["ID"]): objects[
                t["IndexTrackObjectBegin"] : t["IndexTrackObjectEnd"]
            ]
            for t in tracks
        }

    def indices_at(self, t):
        """Object indices present at timepoint ``t``."""
        return np.flatnonzero(self.time_indices == t)

    def octree(self, index) -> SurfaceOctree:
        return _decode_model(self._group, self._models[index])

    def labels(self, t, *, res_level=0, coverage=None):
        """uint32 label image (z, y, x) at timepoint ``t``: object ID + 1, 0 outside.

        See :meth:`SurfaceOctree.rasterize` for ``coverage``.
        """
        grid = self._grid(res_level)
        out = np.zeros(grid.shape, np.uint32)
        for i in self.indices_at(t):
            mask, sl = self.octree(i).rasterize(grid, coverage)
            out[sl][mask] = self.object_ids[i] + 1
        return out

    def mask(self, t, *, res_level=0, coverage=None):
        """Boolean mask (z, y, x) of all objects at timepoint ``t``."""
        grid = self._grid(res_level)
        out = np.zeros(grid.shape, bool)
        for i in self.indices_at(t):
            mask, sl = self.octree(i).rasterize(grid, coverage)
            out[sl] |= mask
        return out

    def mesh(self, index, *, res_level=0) -> SurfaceMesh:
        """Mesh of one object in voxel index coordinates (z, y, x) of ``res_level``.

        Voxel centers sit at integer coordinates, so scaling the vertices by
        the voxel size places the mesh exactly over the image.
        """
        grid = self._grid(res_level)
        m = self.octree(index).mesh()
        return SurfaceMesh(
            vertices=grid.to_voxel(m.vertices),
            faces=m.faces,
            normals=m.normals,
        )


def read_surfaces(h5file, grid_for_level):
    """All Surfaces items of an open .ims file, ``{name: ImarisSurfaces}``."""
    if "Scene8/Content" not in h5file:
        return {}
    content = h5file["Scene8/Content"]
    keys = sorted(
        (k for k in content if k.startswith("MegaSurfaces")),
        key=lambda k: int(k[12:] or 0),
    )
    out = {}
    for key in keys:
        item = ImarisSurfaces(content[key], grid_for_level)
        name = item.name if item.name not in out else f"{item.name} [{key}]"
        out[name] = item
    return out
