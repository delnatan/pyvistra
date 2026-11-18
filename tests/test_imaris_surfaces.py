"""Imaris Surfaces (MegaSurfaces) decoding: readers/imaris_surfaces.py."""

import os

import h5py
import numpy as np
import pytest

from pyvistra.readers import ImarisReader
from pyvistra.readers.imaris_surfaces import (
    CORNER_OFFSETS,
    SurfaceOctree,
    decode_distance,
    decode_normal,
    encode_corners,
)
from pyvistra.readers.imaris_writer import save_imaris

IMARIS_DEMOS = "/Library/Application Support/Bitplane/ImarisImages"

# image: 20 x 40 x 40 voxels of 0.5 x 0.25 x 0.25 um -> 10 um cube
VOXEL = (0.5, 0.25, 0.25)
IMAGE_SHAPE = (20, 40, 40)
# surface lattice like Imaris': xy binned 2x, vertices at binned-voxel centers
LATTICE_ORIGIN = (0.25, 0.25, 0.25)
LATTICE_SPACING = (0.5, 0.5, 0.5)
LATTICE_CELLS = (19, 19, 19)


def sphere_octree(center, radius, *, object_id=0, time_index=0):
    axes = [
        o + np.arange(n + 1) * s
        for o, s, n in zip(LATTICE_ORIGIN, LATTICE_SPACING, LATTICE_CELLS)
    ]
    p = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    sdf = radius - np.linalg.norm(p - center, axis=-1)
    return SurfaceOctree.from_signed_distance(
        sdf,
        LATTICE_ORIGIN,
        LATTICE_SPACING,
        object_id=object_id,
        time_index=time_index,
    )


# --------------------------------------------------------------------------
# Writing octrees in Imaris' layout (the inverse of _decode_model)
# --------------------------------------------------------------------------


def _record(codes=None):
    """32-byte leaf record: 8 x uint16 low halves, then 8 x (byte 2, byte 3)."""
    if codes is None:
        return np.full(32, 0xFF, np.uint8)
    codes = np.asarray(codes, np.uint32)
    rec = np.empty(32, np.uint8)
    rec[:16] = (codes & 0xFFFF).astype("<u2").view(np.uint8)
    rec[16::2] = (codes >> 16) & 0xFF
    rec[17::2] = codes >> 24
    return rec


def _serialize(octree, split_level=None):
    """Blocks of one model: [{parent, nodes, leaves, split}], root block first.

    With ``split_level`` set, that level's nodes go into two sibling blocks
    (and their subtrees stay in separate blocks below it).
    """
    levels = octree.levels
    cells = [
        {
            **{tuple(o): None for o in lv.full},
            **{tuple(o): c for o, c in zip(lv.crossed, lv.codes)},
        }
        for lv in levels
    ]
    blocks = [{"parent": 0, "nodes": [], "leaves": [], "split": []}]
    current = [(0, (0, 0, 0))]  # (block, origin) of the nodes at this depth
    for k, level in enumerate(levels):
        size = level.cell_size
        nxt = []
        for b in sorted({blk for blk, _ in current}):
            mine = [o for blk, o in current if blk == b]
            refined_per_node = []
            for o in mine:
                cmask = dmask = 0
                refined = []
                data = []
                for c in range(8):
                    child = tuple(np.add(o, CORNER_OFFSETS[c] * size))
                    if child not in cells[k]:
                        continue
                    codes = cells[k][child]
                    dmask |= 1 << c
                    data.append(_record(codes))
                    if codes is not None and k < len(levels) - 1:
                        cmask |= 1 << c
                        refined.append(child)
                refined_per_node.append(refined)
                blocks[b]["nodes"].append((cmask, dmask, len(data)))
                blocks[b]["leaves"].extend(data)
            if k == len(levels) - 1 or not any(refined_per_node):
                continue
            ranges = [(0, len(mine))]
            if split_level == k + 1 and len(mine) > 1:
                m = len(mine) // 2
                ranges = [(0, m), (m, len(mine))]
            for lo, hi in ranges:
                blocks.append(
                    {"parent": b, "nodes": [], "leaves": [], "split": []}
                )
                blocks[b]["split"].append(hi)
                nxt += [
                    (len(blocks) - 1, child)
                    for node in refined_per_node[lo:hi]
                    for child in node
                ]
        current = nxt
    # node words: child offsets / data offsets are local to each block
    for blk_i, blk in enumerate(blocks):
        words, n_data = [], 0
        child_counts = {}
        child_blocks = [
            j
            for j, other in enumerate(blocks)
            if j and other["parent"] == blk_i
        ]
        bounds = blk["split"]
        for i, (cmask, dmask, nd) in enumerate(blk["nodes"]):
            target = (
                child_blocks[int(np.searchsorted(bounds, i, side="right"))]
                if cmask
                else None
            )
            coff = child_counts.get(target, 0)
            words.append(
                (coff | cmask << 24) | ((n_data * 8) | dmask << 24) << 32
            )
            if cmask:
                child_counts[target] = coff + bin(cmask).count("1")
            n_data += nd
        blk["words"] = np.array(words, np.uint64)
    return blocks


def write_surfaces(path, items, *, split_level=None, row_width=64):
    """Add Surfaces items {name: [SurfaceOctree, ...]} to an existing .ims file."""
    with h5py.File(path, "a") as f:
        content = f.require_group("Scene8/Content")
        for k, (name, octrees) in enumerate(items.items()):
            g = content.create_group(f"MegaSurfaces{k}")
            g.attrs["Name"] = np.bytes_(name)
            model_rows, block_rows, level_rows, split_rows, flat = (
                [],
                [],
                [],
                [],
                [],
            )
            for oc in octrees:
                b0, l0 = len(block_rows), len(level_rows)
                blocks = _serialize(oc, split_level)
                for blk in blocks:
                    start = sum(len(w) for w in flat)
                    leaves = (
                        np.concatenate(blk["leaves"]).view(np.uint64)
                        if blk["leaves"]
                        else np.zeros(0, np.uint64)
                    )
                    flat += [blk["words"], leaves]
                    s0 = len(split_rows)
                    split_rows += blk["split"]
                    nb, nl = len(blk["words"]), len(leaves)
                    block_rows.append(
                        (
                            blk["parent"],
                            0,
                            0,
                            s0,
                            len(split_rows),
                            start,
                            start + nb,
                            start + nb,
                            start + nb + nl,
                        )
                    )
                level_rows += [(2, 2, 2)] * len(oc.levels)
                lo = np.array(oc.origin)[::-1]
                hi = (np.array(oc.origin) + np.array(oc.shape) * oc.spacing)[
                    ::-1
                ]
                model_rows.append(
                    (
                        oc.time_index,
                        oc.object_id,
                        len(oc.levels),
                        *oc.shape[::-1],
                        *lo,
                        *hi,
                        8,
                        0.5,
                        l0,
                        len(level_rows),
                        b0,
                        len(block_rows),
                    )
                )
            i8, f4 = "<i8", "<f4"
            g["SurfaceModel"] = np.array(
                model_rows,
                dtype=[
                    ("ID_Time", i8),
                    ("ID", i8),
                    ("NumberOfLevels", i8),
                    ("SizeX", i8),
                    ("SizeY", i8),
                    ("SizeZ", i8),
                    ("ExtendMinX", f4),
                    ("ExtendMinY", f4),
                    ("ExtendMinZ", f4),
                    ("ExtendMaxX", f4),
                    ("ExtendMaxY", f4),
                    ("ExtendMaxZ", f4),
                    ("DataCountPerChildNode", i8),
                    ("Threshold", f4),
                    ("LevelInfoBegin", i8),
                    ("LevelInfoEnd", i8),
                    ("BlockInfoBegin", i8),
                    ("BlockInfoEnd", i8),
                ],
            )
            g["BlockInfo"] = np.array(
                block_rows,
                dtype=[
                    (n, i8)
                    for n in (
                        "ParentBlockIndex",
                        "RelativeBlockPathBegin",
                        "RelativeBlockPathEnd",
                        "SplitOffsetsBegin",
                        "SplitOffsetsEnd",
                        "NodeBegin",
                        "NodeEnd",
                        "LeafBegin",
                        "LeafEnd",
                    )
                ],
            )
            g["LevelInfo"] = np.array(
                level_rows,
                dtype=[
                    ("SamplingX", i8),
                    ("SamplingY", i8),
                    ("SamplingZ", i8),
                ],
            )
            g["SplitOffset"] = np.array(
                [(s,) for s in split_rows], dtype=[("Data", i8)]
            )
            data = np.concatenate(flat)
            data = np.concatenate(
                [data, np.zeros(-len(data) % row_width, np.uint64)]
            )
            g["BlockData"] = data.view(np.int64).reshape(-1, row_width)


def make_ims(tmp_path, items, n_times=1, **kw):
    path = str(tmp_path / "surfaces.ims")
    save_imaris(
        path,
        np.zeros(
            (n_times, IMAGE_SHAPE[0], 1, *IMAGE_SHAPE[1:]), np.uint16
        ),  # (T, Z, C, Y, X)
        metadata={"scale": VOXEL},
        resolution_levels=False,
    )
    write_surfaces(path, items, **kw)
    return path


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_corner_codes_round_trip():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(500, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    d = rng.uniform(-1, 1, 500)
    codes = encode_corners(d, n)
    # sqrt encoding: 13 bits resolve sqrt(|d|) to 1/4096
    assert np.allclose(
        decode_distance(codes), d, atol=2 * np.sqrt(np.abs(d)) / 4095 + 1e-6
    )
    assert np.min(np.sum(decode_normal(codes) * n, axis=1)) > 0.9999
    # face bits: +x is face 1, -z is face 6
    assert encode_corners(0.0, [0.0, 0.0, 1.0]) >> 13 & 7 == 1
    assert encode_corners(0.0, [-1.0, 0.0, 0.0]) >> 13 & 7 == 6


def test_sphere_mask_and_mesh_follow_the_surface(tmp_path):
    center, radius = np.array([5.0, 4.9, 5.1]), 3.0
    path = make_ims(
        tmp_path, {"Cell": [sphere_octree(center, radius, object_id=7)]}
    )
    with ImarisReader(path) as r:
        assert list(r.surfaces) == ["Cell"]
        surf = r.surfaces["Cell"]
        assert len(surf) == 1 and surf.object_ids[0] == 7

        mask = surf.mask(0)
        assert mask.shape == IMAGE_SHAPE
        volume = mask.sum() * np.prod(VOXEL)
        assert volume == pytest.approx(4 / 3 * np.pi * radius**3, rel=0.03)
        grid = r.image_grid()
        assert np.allclose(
            grid.to_physical(np.argwhere(mask).mean(0)), center, atol=0.05
        )

        mesh = surf.mesh(0)  # voxel coordinates of the image
        p = grid.to_physical(mesh.vertices)
        assert np.abs(np.linalg.norm(p - center, axis=1) - radius).max() < 0.1
        radial = (p - center) / np.linalg.norm(
            p - center, axis=1, keepdims=True
        )
        assert np.mean(np.sum(mesh.normals * radial, axis=1)) > 0.99

        physical = surf.octree(0).mesh()
        assert physical.volume == pytest.approx(
            4 / 3 * np.pi * radius**3, rel=0.03
        )  # > 0: outward winding
        assert physical.area == pytest.approx(4 * np.pi * radius**2, rel=0.03)


def test_coverage_grows_the_mask_monotonically(tmp_path):
    path = make_ims(tmp_path, {"Cell": [sphere_octree((5.0, 5.0, 5.0), 2.5)]})
    with ImarisReader(path) as r:
        surf = r.surfaces["Cell"]
        counts = [surf.mask(0, coverage=c).sum() for c in (0.75, 0.5, 0.25)]
        assert counts[0] < counts[1] < counts[2]
        # coverage 0.5 and the voxel-center rule agree up to the boundary shell
        assert abs(counts[1] - surf.mask(0).sum()) < 0.1 * counts[1]


def test_multi_block_octree_decodes_like_single_block(tmp_path):
    octree = sphere_octree((5.0, 4.9, 5.1), 3.0)
    (tmp_path / "single").mkdir()
    (tmp_path / "split").mkdir()
    single = make_ims(tmp_path / "single", {"S": [octree]})
    split = make_ims(tmp_path / "split", {"S": [octree]}, split_level=3)
    with h5py.File(split, "r") as f:
        assert len(f["Scene8/Content/MegaSurfaces0/BlockInfo"]) > len(
            octree.levels
        )
    with ImarisReader(single) as a, ImarisReader(split) as b:
        oa, ob = a.surfaces["S"].octree(0), b.surfaces["S"].octree(0)
    for la, lb in zip(oa.levels, ob.levels):
        assert sorted(map(tuple, la.full)) == sorted(map(tuple, lb.full))
        order_a, order_b = np.lexsort(la.crossed.T), np.lexsort(lb.crossed.T)
        assert np.array_equal(la.crossed[order_a], lb.crossed[order_b])
        assert np.array_equal(la.codes[order_a], lb.codes[order_b])
    assert np.array_equal(oa.lattice, octree.lattice)


def test_labels_hold_object_ids_per_timepoint(tmp_path):
    items = {
        "Cells": [
            sphere_octree((3.0, 3.0, 3.0), 1.5, object_id=0, time_index=0),
            sphere_octree((7.0, 7.0, 7.0), 1.5, object_id=4, time_index=0),
            sphere_octree((5.0, 5.0, 5.0), 2.0, object_id=9, time_index=1),
        ]
    }
    path = make_ims(tmp_path, items, n_times=2)
    with ImarisReader(path) as r:
        surf = r.surfaces["Cells"]
        assert list(surf.indices_at(0)) == [0, 1]
        assert set(np.unique(surf.labels(0))) == {0, 1, 5}
        assert set(np.unique(surf.labels(1))) == {0, 10}
        assert (
            surf.labels(0)[6, 12, 12] == 1
        )  # (3, 3, 3) um -> voxel (5.5, 11.5, 11.5)


def test_files_without_surfaces_have_none(tmp_path):
    path = str(tmp_path / "plain.ims")
    save_imaris(
        path,
        np.zeros((1, 4, 1, 8, 8), np.uint16),
        metadata={"scale": (1, 1, 1)},
        resolution_levels=False,
    )
    with ImarisReader(path) as r:
        assert r.surfaces == {}


@pytest.mark.parametrize(
    "name, exact_triangles",
    [
        ("PtK2Cell_with_objects.ims", True),
        (
            "Microglia-vasculature_with_objects.ims",
            False,
        ),  # multi-block levels
    ],
)
def test_meshes_match_imaris_statistics(name, exact_triangles):
    path = os.path.join(IMARIS_DEMOS, name)
    if not os.path.exists(path):
        pytest.skip(f"Imaris demo image {name} not installed")
    with ImarisReader(path) as r:
        for surf in r.surfaces.values():
            stats = surf.statistics
            for i in range(0, len(surf), max(1, len(surf) // 8)):
                mesh = surf.octree(i).mesh()
                if exact_triangles:
                    assert len(mesh.faces) == stats["NumberOfTriangles"][i]
                assert mesh.area == pytest.approx(stats["Area"][i], rel=2e-3)
                assert mesh.volume == pytest.approx(
                    stats["Volume"][i], rel=1e-2
                )
