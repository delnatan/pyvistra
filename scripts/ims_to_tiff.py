#!/usr/bin/env python
"""
Convert Imaris (.ims) files to 3D TIFF stacks, one stack per channel.

Self-contained: only needs `h5py`, `numpy`, and `tifffile`
    pip install h5py numpy tifffile

Usage
-----
    python ims_to_tiff.py  INPUT_DIR  OUTPUT_DIR  [options]

Example (Windows):
    python ims_to_tiff.py D:\\scan\\ims D:\\scan\\tiff

Output layout (default, `--layout channel-dirs`):
    OUTPUT_DIR/
        C0/<file>.tif        # ZYX stack, channel 0
        C1/<file>.tif
        ...
        tile_positions.csv   # per-file stage extents in microns

Per-channel folders are convenient for TeraStitcher, which stitches one
channel at a time: point each run at a different `C<n>` folder (or stitch
one channel and merge the others using the same computed displacements).

With `--layout flat`, files are written as OUTPUT_DIR/<file>_C0.tif, etc.
Time series get a `_T000` suffix per timepoint.

Planes are streamed from HDF5 to TIFF one Z-slice at a time, so memory use
stays at roughly one plane regardless of stack size.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import tifffile

# ImageJ hyperstacks only support these dtypes and must stay under 4 GB;
# anything else is written as a plain (Big)TIFF instead.
IMAGEJ_DTYPES = {np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")}
IMAGEJ_MAX_BYTES = 2**32 - 2**25  # leave headroom for tags/metadata


# --------------------------------------------------------------------------
# Imaris reader (trimmed from pyvistra.readers.imaris)
# --------------------------------------------------------------------------
class ImarisReader:
    """Minimal reader for Imaris .ims (HDF5) files.

    shape is (T, C, Z, Y, X); voxel_size is (Z, Y, X) in microns;
    extent_min / extent_max are (X, Y, Z) stage extents in microns.
    """

    def __init__(self, filepath):
        self.filepath = str(filepath)
        self._file = h5py.File(self.filepath, "r")
        self._scan_structure()
        self._parse_metadata()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    @staticmethod
    def _decode(value):
        """Imaris stores strings as arrays of single bytes: [b'6', b'0']."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        arr = np.asarray(value).ravel()
        if arr.size == 0:
            return ""
        if arr.dtype.kind == "S":
            return b"".join(arr.tolist()).decode("utf-8", errors="replace")
        if arr.dtype.kind == "U":
            return "".join(arr.tolist())
        if arr.dtype.kind in "iu":
            return "".join(chr(int(x)) for x in arr)
        return str(arr[0])

    def _attr(self, grp, key, cast=str, default=None):
        if grp is None or key not in grp.attrs:
            return default
        s = self._decode(grp.attrs[key]).strip()
        if not s:
            return default
        try:
            return cast(s)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _child(parent, base, index):
        """Find `base <index>` or `base<index>` in an HDF5 group."""
        for name in (f"{base} {index}", f"{base}{index}"):
            if name in parent:
                return parent[name]
        pat = re.compile(rf"^{re.escape(base)}\s*(\d+)$")
        for name in parent.keys():
            m = pat.match(name)
            if m and int(m.group(1)) == index:
                return parent[name]
        return None

    def _logical_size(self, c_grp, data):
        """ImageSize attrs give the real size; Data may be chunk-padded."""
        sizes = []
        for key, axis in (("ImageSizeZ", 0), ("ImageSizeY", 1), ("ImageSizeX", 2)):
            v = self._attr(c_grp, key, int)
            if v is None:
                v = self._attr(data, key, int)
            if v is None:
                v = data.shape[axis]
            sizes.append(min(int(v), data.shape[axis]))
        return tuple(sizes)

    def _scan_structure(self):
        root = self._file["DataSet"]
        res = sorted(
            (k for k in root.keys() if re.match(r"ResolutionLevel\s*\d+$", k)),
            key=lambda k: int(re.search(r"\d+", k).group()),
        )
        if not res:
            raise ValueError("No ResolutionLevels found")
        self._res_groups = res

        l0 = root[res[0]]
        self.n_timepoints = sum(1 for k in l0.keys() if k.startswith("TimePoint"))
        t0 = self._child(l0, "TimePoint", 0)
        if t0 is None:
            raise ValueError("TimePoint 0 not found")
        self.n_channels = sum(1 for k in t0.keys() if k.startswith("Channel"))
        c0 = self._child(t0, "Channel", 0)
        if c0 is None or "Data" not in c0:
            raise ValueError("Channel 0/Data not found")

        data = c0["Data"]
        self.dtype = data.dtype
        self.size_z, self.size_y, self.size_x = self._logical_size(c0, data)
        self.shape = (
            self.n_timepoints,
            self.n_channels,
            self.size_z,
            self.size_y,
            self.size_x,
        )

    def _parse_metadata(self):
        info = self._file.get("DataSetInfo")
        img = info.get("Image") if info is not None else None

        self.extent_min = tuple(
            self._attr(img, f"ExtMin{i}", float, 0.0) for i in range(3)
        )
        self.extent_max = tuple(
            self._attr(img, f"ExtMax{i}", float, float(n))
            for i, n in enumerate((self.size_x, self.size_y, self.size_z))
        )
        self.unit = self._attr(img, "Unit", str, "um")

        sizes_xyz = (self.size_x, self.size_y, self.size_z)
        vx, vy, vz = (
            (hi - lo) / n if n > 0 else 1.0
            for lo, hi, n in zip(self.extent_min, self.extent_max, sizes_xyz)
        )
        self.voxel_size = (abs(vz), abs(vy), abs(vx))

        self.channel_names = []
        for c in range(self.n_channels):
            grp = info.get(f"Channel {c}") if info is not None else None
            self.channel_names.append(self._attr(grp, "Name", str, f"Channel {c}"))

    def dataset(self, c=0, t=0, res_level=0):
        """Return (h5py dataset, (Z, Y, X) logical size) for one volume."""
        if res_level >= len(self._res_groups):
            raise ValueError(
                f"Resolution level {res_level} unavailable "
                f"(file has {len(self._res_groups)})"
            )
        res = self._file["DataSet"][self._res_groups[res_level]]
        t_grp = self._child(res, "TimePoint", t)
        c_grp = self._child(t_grp, "Channel", c) if t_grp is not None else None
        if c_grp is None or "Data" not in c_grp:
            raise ValueError(f"Volume t={t} c={c} not found")
        data = c_grp["Data"]
        return data, self._logical_size(c_grp, data)


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def write_stack(path, dset, size_zyx, voxel_size_zyx, compress):
    """Stream a ZYX volume from HDF5 into a multipage TIFF."""
    nz, ny, nx = size_zyx
    dtype = dset.dtype
    vz, vy, vx = voxel_size_zyx
    planes = (dset[z, :ny, :nx] for z in range(nz))

    nbytes = nz * ny * nx * dtype.itemsize
    use_imagej = dtype in IMAGEJ_DTYPES and nbytes < IMAGEJ_MAX_BYTES and not compress

    kwargs = dict(
        shape=(nz, ny, nx),
        dtype=dtype,
        resolution=(1.0 / vx, 1.0 / vy) if vx and vy else None,
    )
    if use_imagej:
        # ImageJ/Fiji reads spacing + unit from this; TeraStitcher just
        # sees a normal multipage TIFF.
        kwargs.update(
            imagej=True,
            metadata={"axes": "ZYX", "spacing": vz, "unit": "um"},
        )
    else:
        kwargs.update(
            bigtiff=nbytes >= IMAGEJ_MAX_BYTES,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        if compress:
            kwargs["compression"] = "zlib"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tifffile.imwrite(tmp, planes, **kwargs)
    tmp.replace(path)  # never leave a half-written .tif behind


def output_path(out_dir, stem, c, t, n_t, layout):
    t_tag = f"_T{t:03d}" if n_t > 1 else ""
    if layout == "channel-dirs":
        return out_dir / f"C{c}" / f"{stem}{t_tag}.tif"
    return out_dir / f"{stem}{t_tag}_C{c}.tif"


def convert_file(ims_path, out_dir, args):
    with ImarisReader(ims_path) as r:
        print(
            f"  shape (T,C,Z,Y,X)={r.shape} dtype={r.dtype} "
            f"voxel(Z,Y,X)={tuple(round(v, 4) for v in r.voxel_size)} {r.unit}"
        )
        print(f"  channels: {r.channel_names}")

        channels = args.channels if args.channels else range(r.n_channels)
        for t in range(r.n_timepoints):
            for c in channels:
                if c >= r.n_channels:
                    print(f"  ! channel {c} not in file, skipping")
                    continue
                dst = output_path(
                    out_dir, ims_path.stem, c, t, r.n_timepoints, args.layout
                )
                if dst.exists() and not args.overwrite:
                    print(f"  = exists, skipping {dst}")
                    continue
                dset, size = r.dataset(c=c, t=t, res_level=args.res_level)
                scale = [
                    full / sub
                    for full, sub in zip((r.size_z, r.size_y, r.size_x), size)
                ]
                voxel = tuple(v * s for v, s in zip(r.voxel_size, scale))
                t0 = time.perf_counter()
                write_stack(dst, dset, size, voxel, args.compress)
                print(
                    f"  -> {dst}  ({time.perf_counter() - t0:.1f}s)"
                )

        return {
            "file": ims_path.name,
            "min_x_um": r.extent_min[0],
            "min_y_um": r.extent_min[1],
            "min_z_um": r.extent_min[2],
            "max_x_um": r.extent_max[0],
            "max_y_um": r.extent_max[1],
            "max_z_um": r.extent_max[2],
            "size_x": r.size_x,
            "size_y": r.size_y,
            "size_z": r.size_z,
            "voxel_x_um": r.voxel_size[2],
            "voxel_y_um": r.voxel_size[1],
            "voxel_z_um": r.voxel_size[0],
            "channels": ";".join(r.channel_names),
        }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert Imaris .ims files to per-channel 3D TIFF stacks.",
    )
    p.add_argument("input_dir", type=Path, help="folder containing .ims files")
    p.add_argument("output_dir", type=Path, help="folder to write TIFFs into")
    p.add_argument(
        "--layout",
        choices=["channel-dirs", "flat"],
        default="channel-dirs",
        help="channel-dirs: OUT/C0/name.tif (default); flat: OUT/name_C0.tif",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        help="only export these channel indices (default: all)",
    )
    p.add_argument(
        "--res-level",
        type=int,
        default=0,
        help="Imaris resolution level (0 = full resolution)",
    )
    p.add_argument(
        "-r", "--recursive", action="store_true", help="search subfolders too"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="overwrite existing TIFFs"
    )
    p.add_argument(
        "--compress",
        action="store_true",
        help="zlib-compress (smaller files, slower; written as plain TIFF)",
    )
    args = p.parse_args(argv)

    if not args.input_dir.is_dir():
        p.error(f"input folder not found: {args.input_dir}")

    pattern = "**/*" if args.recursive else "*"
    files = sorted(
        f for f in args.input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() == ".ims"
    )
    if not files:
        print(f"No .ims files found in {args.input_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f}")
        # Mirror input subfolders so same-named files don't collide.
        out_dir = args.output_dir / f.parent.relative_to(args.input_dir)
        try:
            rows.append(convert_file(f, out_dir, args))
        except Exception as e:  # keep going; report at the end
            print(f"  ! FAILED: {e}")
            failed.append(f)

    if rows:
        csv_path = args.output_dir / "tile_positions.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {csv_path}")

    print(f"Done: {len(rows)} converted, {len(failed)} failed.")
    for f in failed:
        print(f"  failed: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
