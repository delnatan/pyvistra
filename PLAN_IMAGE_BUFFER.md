# Implementation Plan: ImageBuffer and ROI Region Extraction

## Overview

This plan introduces:
1. **ImageBuffer** - Zarr 3-backed array for streaming writes
2. **ROI region extraction** - `get_region()` methods on ROI classes
3. **WYSIWYG transforms** - "Apply Transform" bakes rotation/translation into data

## Design Principles

- **Keep it simple** - No caching, no generic frameworks, no factory methods
- **WYSIWYG** - What you see is what you get; ROI extraction matches display
- **Explicit over magic** - User clicks "Apply" to commit transforms

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WYSIWYG Workflow                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Load Image              2. Preview Transform      3. Apply          │
│  ┌─────────────┐           ┌─────────────┐          ┌─────────────┐    │
│  │ source.ims  │  visual   │ GPU rotate  │  bake    │ ImageBuffer │    │
│  │ (proxy)     │──────────►│ (instant)   │─────────►│ (Zarr)      │    │
│  └─────────────┘  only     └─────────────┘  data    └──────┬──────┘    │
│                                                            │           │
│  4. Draw ROI                5. Extract Region              │           │
│  ┌─────────────┐           ┌─────────────┐                │           │
│  │ Rectangle   │───────────│ roi.get_    │◄───────────────┘           │
│  │ on image    │  matches  │ region()    │  from buffer               │
│  └─────────────┘  display  └─────────────┘  (transformed)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Add zarr dependency

**File: `pyproject.toml`**
```python
dependencies = [
    # ... existing ...
    "zarr>=3.0",
]
```

### Step 2: ImageBuffer class

**File: `io.py`**

```python
import zarr
from pathlib import Path
import uuid

BUFFER_DIR = Path.home() / '.pyvistra' / 'buffers'


class ImageBuffer:
    """
    Zarr-backed 5D array buffer for streaming image operations.

    Same interface as Numpy5DProxy for reading, plus write support.
    """

    def __init__(self, shape, dtype, chunks=None, metadata=None):
        """
        Create a new buffer.

        Args:
            shape: 5D shape (T, Z, C, Y, X)
            dtype: numpy dtype
            chunks: Chunk shape, default (1, 16, C, 512, 512)
            metadata: Optional dict to preserve
        """
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        self._path = BUFFER_DIR / f"{uuid.uuid4()}.zarr"

        T, Z, C, Y, X = shape
        if chunks is None:
            chunks = (1, min(16, Z), C, min(512, Y), min(512, X))

        self._store = zarr.open(
            str(self._path),
            mode='w',
            shape=shape,
            dtype=dtype,
            chunks=chunks,
        )

        self.metadata = metadata or {}
        self.ndim = 5

    @property
    def shape(self):
        return self._store.shape

    @property
    def dtype(self):
        return self._store.dtype

    def __getitem__(self, key):
        """Read slices - same interface as proxies."""
        return self._store[key]

    def __setitem__(self, key, value):
        """Write slices."""
        self._store[key] = value

    def save_as(self, filepath):
        """Export buffer to OME-TIFF."""
        from .io import save_tiff
        scale = self.metadata.get('scale', (1.0, 1.0, 1.0))
        save_tiff(filepath, self._store[:], scale=scale)

    def close(self):
        """Close and delete the temporary buffer file."""
        import shutil
        if self._path.exists():
            shutil.rmtree(self._path)

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
```

### Step 3: Transform application

**File: `io.py`** (add function)

```python
def apply_transform(source, rotation_deg, translate, progress_cb=None):
    """
    Apply 2D rotation and translation to create a new buffer.

    Args:
        source: Source proxy (any 5D array-like)
        rotation_deg: Rotation angle in degrees
        translate: (tx, ty) translation in pixels
        progress_cb: Optional callback(progress_fraction)

    Returns:
        ImageBuffer with transformed data
    """
    from scipy.ndimage import affine_transform
    import numpy as np

    T, Z, C, Y, X = source.shape

    # Create output buffer
    buffer = ImageBuffer(
        shape=source.shape,
        dtype=source.dtype,
        metadata=getattr(source, 'metadata', {}),
    )

    # Build affine transform matrix (rotation around center + translation)
    cx, cy = X / 2, Y / 2
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tx, ty = translate

    # Inverse mapping matrix for scipy
    matrix = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    offset = np.array([
        cy - cos_t * cy - sin_t * cx - ty,
        cx + sin_t * cy - cos_t * cx - tx
    ])

    total = T * Z
    for t in range(T):
        for z in range(Z):
            slice_data = source[t, z, :, :, :]  # (C, Y, X)

            # Transform each channel
            transformed = np.stack([
                affine_transform(slice_data[c], matrix, offset, order=1)
                for c in range(C)
            ])

            buffer[t, z, :, :, :] = transformed

            if progress_cb:
                progress_cb((t * Z + z + 1) / total)

    return buffer
```

### Step 4: Update TransformDialog

**File: `widgets.py`** (modify TransformDialog)

Add an "Apply" button that bakes the transform:

```python
class TransformDialog(QDialog):
    def __init__(self, image_window, parent=None):
        # ... existing setup ...

        # Add Apply button
        self.apply_btn = QPushButton("Apply Transform")
        self.apply_btn.clicked.connect(self.apply_transform)
        self.layout.addWidget(self.apply_btn)

    def apply_transform(self):
        """Bake current rotation/translation into image data."""
        from .io import apply_transform

        rotation = self.rotation_slider.value()
        tx = self.translate_x.value()
        ty = self.translate_y.value()

        # Skip if no transform
        if rotation == 0 and tx == 0 and ty == 0:
            return

        # Show progress (optional: could add progress dialog)
        self.apply_btn.setEnabled(False)
        self.apply_btn.setText("Applying...")
        QApplication.processEvents()

        try:
            # Create transformed buffer
            buffer = apply_transform(
                self.window.img_data,
                rotation,
                (tx, ty),
            )
            buffer.metadata = self.window.meta.copy()

            # Switch window to use buffer
            self.window.img_data = buffer
            self.window.renderer.data = buffer
            self.window.meta = buffer.metadata

            # Reset visual transform (data is now transformed)
            self.window.renderer.set_rotation(0)
            self.window.renderer.set_translation(0, 0)
            self.rotation_slider.setValue(0)
            self.translate_x.setValue(0)
            self.translate_y.setValue(0)

            # Refresh display
            self.window.update_view()

        finally:
            self.apply_btn.setEnabled(True)
            self.apply_btn.setText("Apply Transform")
```

### Step 5: ROI region extraction

**File: `rois.py`** (add methods to ROI classes)

```python
class RectangleROI(ROI):
    # ... existing code ...

    def get_region(self, data):
        """
        Extract rectangular region from data.

        Args:
            data: Array with shape (..., Y, X)

        Returns:
            Cropped array with shape (..., height, width)
        """
        x1, y1 = self.data['p1']
        x2, y2 = self.data['p2']

        # Normalize to min/max
        xmin, xmax = int(min(x1, x2)), int(max(x1, x2))
        ymin, ymax = int(min(y1, y2)), int(max(y1, y2))

        # Clamp to bounds
        Y, X = data.shape[-2:]
        xmin, xmax = max(0, xmin), min(X, xmax)
        ymin, ymax = max(0, ymin), min(Y, ymax)

        return data[..., ymin:ymax, xmin:xmax]


class CircleROI(ROI):
    # ... existing code ...

    def get_region(self, data):
        """
        Extract circular region from data.

        Args:
            data: Array with shape (..., Y, X)

        Returns:
            tuple: (region, mask) where region is bounding box
                   and mask is boolean array for circle
        """
        import numpy as np

        cx, cy = self.data['center']
        ex, ey = self.data['edge']
        radius = np.sqrt((ex - cx)**2 + (ey - cy)**2)

        # Bounding box
        xmin, xmax = int(cx - radius), int(cx + radius + 1)
        ymin, ymax = int(cy - radius), int(cy + radius + 1)

        # Clamp to bounds
        Y, X = data.shape[-2:]
        xmin, xmax = max(0, xmin), min(X, xmax)
        ymin, ymax = max(0, ymin), min(Y, ymax)

        region = data[..., ymin:ymax, xmin:xmax]

        # Create circular mask
        h, w = ymax - ymin, xmax - xmin
        yy, xx = np.ogrid[:h, :w]
        local_cx, local_cy = cx - xmin, cy - ymin
        mask = ((xx - local_cx)**2 + (yy - local_cy)**2) <= radius**2

        return region, mask


class LineROI(ROI):
    # ... existing code ...

    def get_profile(self, data, num_points=None):
        """
        Extract intensity profile along line.

        Args:
            data: Array with shape (..., Y, X)
            num_points: Number of samples (default: line length)

        Returns:
            Array with shape (..., num_points)
        """
        import numpy as np
        from scipy.ndimage import map_coordinates

        x1, y1 = self.data['p1']
        x2, y2 = self.data['p2']

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if num_points is None:
            num_points = max(2, int(np.ceil(length)))

        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)
        coords = np.array([ys, xs])  # scipy uses (row, col) order

        # Handle multi-dimensional data
        if data.ndim == 2:
            return map_coordinates(data, coords, order=1)
        else:
            # For (C, Y, X) or similar, extract per channel
            result = []
            for i in range(data.shape[0]):
                result.append(map_coordinates(data[i], coords, order=1))
            return np.stack(result)
```

## File Changes Summary

| File | Change |
|------|--------|
| `pyproject.toml` | Add `zarr>=3.0` |
| `io.py` | Add `ImageBuffer` class, `apply_transform()` function |
| `widgets.py` | Add "Apply Transform" button to `TransformDialog` |
| `rois.py` | Add `get_region()` to Rectangle/Circle, `get_profile()` to Line |

## Usage Examples

### Transform and crop workflow
```python
# 1. Open image, rotate to align features
window = ImageWindow("sample.ims")
window.show()

# 2. User adjusts rotation in Transform dialog, clicks "Apply"
#    (transforms data, resets visual rotation)

# 3. Draw rectangle ROI on aligned image

# 4. Extract region (matches what user sees)
roi = window.rois[0]
cache = window.renderer.current_slice_cache  # (C, Y, X)
cropped = roi.get_region(cache)
```

### Circle ROI with mask
```python
roi = window.rois[0]  # CircleROI
region, mask = roi.get_region(cache)

# Get mean intensity inside circle
mean_per_channel = [region[c][mask].mean() for c in range(region.shape[0])]
```

### Line profile
```python
roi = window.rois[0]  # LineROI
profile = roi.get_profile(cache)  # shape (C, num_points)
```

---

*Ready for implementation.*
