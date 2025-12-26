# Implementation Plan: ImageBuffer & Streaming Image Operations

## Overview

This plan introduces an `ImageBuffer` system that enables:
1. **Streaming writes** - Process large images slice-by-slice without loading into memory
2. **Live preview** - View processing results in real-time
3. **Composable processing** - Any numpy-compatible function can process data
4. **Transform persistence** - Save rotated/translated images properly

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Data Flow Architecture                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Source Files              ImageBuffer (Zarr 3)              Export          │
│  ┌─────────────┐          ┌─────────────────────┐         ┌─────────────┐   │
│  │ .ims        │──────────│ ~/.pyvistra/        │─────────│ .ome.tif    │   │
│  │ .tif        │  convert │   buffers/          │  save   │ .ims        │   │
│  │ .png        │──────────│     <uuid>.zarr     │─────────│ .ome.zarr   │   │
│  │ numpy array │          └──────────┬──────────┘         └─────────────┘   │
│  └─────────────┘                     │                                       │
│                                      │ Proxy interface                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         ImageWindow                                   │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │   │
│  │  │ CompositeVisual │◄───│ current_slice   │◄───│ buffer[t,z,:,:,:]   │   │
│  │  │ (GPU rendering) │    │ _cache          │    │                 │   │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Processing Pipeline (any framework: numpy, scipy, pytorch, jax, mlx)        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  for t, z in source:                                                  │   │
│  │      slice = source[t, z, :, :, :]     # Read from source             │   │
│  │      result = process_fn(slice)         # Any array operation         │   │
│  │      buffer.write_slice(t, z, result)   # Write to buffer             │   │
│  │      yield progress                     # Live update                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Buffer format | Zarr 3 | Concurrent R/W, chunked, modern |
| Buffer location | `~/.pyvistra/buffers/` | Avoids external drive issues |
| Default chunks | `(1, 16, C, 512, 512)` | Configurable, good default |
| Metadata | Preserve from source | Scientific reproducibility |
| Cleanup | On ImageWindow close | Prevents disk bloat |

## Implementation Steps

### Phase 1: Core Infrastructure

#### Step 1.1: Add zarr dependency
- Update `pyproject.toml` to include `zarr>=3.0`

#### Step 1.2: Create `ImageBuffer` class in `io.py`

```python
class ImageBuffer:
    """
    Zarr-backed buffer for streaming 5D image operations.

    Implements the same slicing interface as Numpy5DProxy/Imaris5DProxy,
    allowing seamless use with ImageWindow.
    """

    def __init__(
        self,
        shape: tuple,
        dtype: np.dtype,
        chunks: tuple = None,  # Default: (1, 16, C, 512, 512)
        path: str = None,      # None = auto-generate in ~/.pyvistra/
        metadata: dict = None,
        compressor: str = 'zstd',
    ):
        """Create or open an ImageBuffer."""

    # Proxy interface (matches Numpy5DProxy)
    @property
    def shape(self) -> tuple: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def ndim(self) -> int: ...

    def __getitem__(self, key) -> np.ndarray:
        """Read slices - same interface as other proxies."""

    # Write interface
    def write_slice(self, t: int, z: int, data: np.ndarray):
        """Write a (C, Y, X) slice at position (t, z)."""

    def write_region(self, region: tuple, data: np.ndarray):
        """Write arbitrary region: buffer[region] = data."""

    # Factory methods
    @classmethod
    def from_proxy(cls, proxy, metadata=None, progress_cb=None) -> 'ImageBuffer':
        """Create buffer by copying from existing proxy (lazy load)."""

    @classmethod
    def from_file(cls, filepath: str) -> 'ImageBuffer':
        """Open existing buffer file."""

    @classmethod
    def empty_like(cls, proxy, metadata=None) -> 'ImageBuffer':
        """Create empty buffer with same shape/dtype as proxy."""

    # Export
    def save_as(self, filepath: str, format: str = 'ome-tiff'):
        """Export buffer to final format."""

    # Lifecycle
    def flush(self):
        """Force write pending chunks to disk."""

    def close(self):
        """Close buffer and optionally delete temp file."""

    def delete(self):
        """Delete buffer file from disk."""
```

#### Step 1.3: Buffer directory management

```python
# In io.py
BUFFER_DIR = Path.home() / '.pyvistra' / 'buffers'

def get_buffer_dir() -> Path:
    """Ensure buffer directory exists and return path."""
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    return BUFFER_DIR

def cleanup_old_buffers(max_age_hours: int = 24):
    """Remove buffer files older than max_age_hours."""
```

### Phase 2: ImageWindow Integration

#### Step 2.1: Dual-mode data handling

ImageWindow should work with both:
- **Direct proxies** (current behavior, read-only)
- **ImageBuffer** (new, read-write)

```python
class ImageWindow:
    def __init__(self, data_or_path, title="Image"):
        # ... existing code ...

        # New: track if we have a buffer
        self._buffer = None  # ImageBuffer or None

        # Load data (unchanged)
        if isinstance(data_or_path, str):
            self.img_data, self.meta = load_image(filepath)
        elif isinstance(data_or_path, ImageBuffer):
            self._buffer = data_or_path
            self.img_data = data_or_path  # Buffer IS the proxy
            self.meta = data_or_path.metadata
        else:
            self.img_data = normalize_to_5d(data_or_path)

    @property
    def buffer(self) -> Optional[ImageBuffer]:
        """Return buffer if window is using one, else None."""
        return self._buffer

    def ensure_buffer(self) -> ImageBuffer:
        """Convert current data to buffer if not already buffered."""
        if self._buffer is None:
            self._buffer = ImageBuffer.from_proxy(self.img_data, self.meta)
            self.img_data = self._buffer
            self.renderer.data = self._buffer
        return self._buffer
```

#### Step 2.2: Transform saving with rotation

Add menu action to save transformed image:

```python
# In ImageWindow._setup_menu()
save_transformed_action = QAction("Save Transformed...", self)
save_transformed_action.triggered.connect(self.save_transformed)
image_menu.addAction(save_transformed_action)

def save_transformed(self):
    """Save current image with rotation/translation applied."""
    filepath, _ = QFileDialog.getSaveFileName(
        self, "Save Transformed Image", "",
        "OME-TIFF (*.ome.tif);;TIFF (*.tif)"
    )
    if not filepath:
        return

    # Get current transform
    rotation = self.renderer._rotation_deg
    tx, ty = self.renderer._translate_x, self.renderer._translate_y

    # Create output buffer
    output = ImageBuffer.empty_like(self.img_data, self.meta)

    # Apply transform slice-by-slice with progress
    apply_transform_to_buffer(
        source=self.img_data,
        output=output,
        rotation_deg=rotation,
        translate=(tx, ty),
        progress_cb=self._update_progress,
    )

    # Export
    output.save_as(filepath)
    output.delete()  # Clean up temp buffer
```

### Phase 3: Processing Pipeline

#### Step 3.1: Generic processing function

```python
# In io.py or new processing.py

def process_image(
    source,           # Any proxy-like object
    process_fn,       # Callable[[np.ndarray], np.ndarray]
    output=None,      # ImageBuffer or None (auto-create)
    slice_dims='tz',  # Which dims to iterate: 'tz', 't', 'z', 'tzc'
    progress_cb=None, # Callable[[float], None]
) -> ImageBuffer:
    """
    Apply process_fn to each slice of source, writing to output buffer.

    Example:
        from scipy.ndimage import gaussian_filter

        smoothed = process_image(
            source=window.img_data,
            process_fn=lambda s: gaussian_filter(s, sigma=2),
        )
        new_window = ImageWindow(smoothed)
    """
```

#### Step 3.2: Transform-specific helper

```python
def apply_transform_to_buffer(
    source,
    output: ImageBuffer,
    rotation_deg: float = 0,
    translate: tuple = (0, 0),
    interpolation_order: int = 1,  # 1=bilinear, 3=bicubic
    progress_cb=None,
):
    """
    Apply 2D rotation and translation to each T/Z slice.

    Uses scipy.ndimage.affine_transform for proper interpolation.
    """
    from scipy.ndimage import affine_transform

    T, Z, C, Y, X = source.shape
    total = T * Z

    # Build transform matrix
    # Rotation around image center + translation
    cx, cy = X / 2, Y / 2
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tx, ty = translate

    # Affine matrix for scipy (inverse mapping)
    matrix = np.array([
        [cos_t, sin_t],
        [-sin_t, cos_t]
    ])
    offset = np.array([
        cy - cos_t * cy - sin_t * cx - ty,
        cx + sin_t * cy - cos_t * cx - tx
    ])

    for t in range(T):
        for z in range(Z):
            slice_data = source[t, z, :, :, :]  # (C, Y, X)

            # Transform each channel
            transformed = np.stack([
                affine_transform(
                    slice_data[c], matrix, offset,
                    order=interpolation_order,
                    mode='constant', cval=0
                )
                for c in range(C)
            ])

            output.write_slice(t, z, transformed)

            if progress_cb:
                progress_cb((t * Z + z + 1) / total)
```

### Phase 4: ROI Region Extraction

With the buffer system in place, ROI region extraction becomes straightforward:

#### Step 4.1: Add `get_region` to ROI classes

```python
# In rois.py

class RectangleROI(ROI):
    def get_region(self, data: np.ndarray) -> np.ndarray:
        """
        Extract region from data array.

        Args:
            data: Array with shape (..., Y, X) - typically (C, Y, X)

        Returns:
            Cropped region with shape (..., height, width)
        """
        x1, y1 = self.data['p1']
        x2, y2 = self.data['p2']

        # Normalize coordinates
        xmin, xmax = int(min(x1, x2)), int(max(x1, x2))
        ymin, ymax = int(min(y1, y2)), int(max(y1, y2))

        # Clamp to data bounds
        Y, X = data.shape[-2:]
        xmin, xmax = max(0, xmin), min(X, xmax)
        ymin, ymax = max(0, ymin), min(Y, ymax)

        return data[..., ymin:ymax, xmin:xmax]


class CircleROI(ROI):
    def get_region(self, data: np.ndarray) -> tuple:
        """
        Extract circular region from data.

        Returns:
            (region, mask): Region is bounding box, mask is boolean circle
        """
        cx, cy = self.data['center']
        ex, ey = self.data['edge']
        radius = np.sqrt((ex - cx)**2 + (ey - cy)**2)

        # Bounding box
        xmin, xmax = int(cx - radius), int(cx + radius)
        ymin, ymax = int(cy - radius), int(cy + radius)

        # Clamp
        Y, X = data.shape[-2:]
        xmin, xmax = max(0, xmin), min(X, xmax)
        ymin, ymax = max(0, ymin), min(Y, ymax)

        region = data[..., ymin:ymax, xmin:xmax]

        # Create circular mask
        h, w = region.shape[-2:]
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - (cx - xmin))**2 + (yy - (cy - ymin))**2) <= radius**2

        return region, mask


class LineROI(ROI):
    def get_profile(self, data: np.ndarray, num_points: int = None) -> np.ndarray:
        """
        Extract intensity profile along line.

        Args:
            data: Array with shape (..., Y, X)
            num_points: Number of sample points (default: line length)

        Returns:
            Profile with shape (..., num_points)
        """
        from scipy.ndimage import map_coordinates

        x1, y1 = self.data['p1']
        x2, y2 = self.data['p2']

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if num_points is None:
            num_points = int(np.ceil(length))

        # Sample coordinates along line
        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)

        # Extract profile for each leading dimension
        coords = np.array([ys, xs])  # scipy uses (y, x) order

        if data.ndim == 2:
            return map_coordinates(data, coords, order=1)
        else:
            # Handle (C, Y, X) or similar
            profiles = []
            for i in range(data.shape[0]):
                profiles.append(map_coordinates(data[i], coords, order=1))
            return np.stack(profiles)
```

#### Step 4.2: Convenience method on ImageWindow

```python
class ImageWindow:
    def get_roi_data(self, roi, source='cache'):
        """
        Get data for an ROI.

        Args:
            roi: ROI instance
            source: 'cache' (current slice), 'full' (all T/Z), or specific indices

        Returns:
            Data array or (data, mask) for CircleROI
        """
        if source == 'cache':
            data = self.renderer.current_slice_cache  # (C, Y, X)
        elif source == 'full':
            data = self.img_data[:]  # Load full array
        else:
            data = self.img_data[source]  # Custom slicing

        if isinstance(roi, LineROI):
            return roi.get_profile(data)
        else:
            return roi.get_region(data)
```

## File Changes Summary

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `zarr>=3.0` dependency |
| `io.py` | Add `ImageBuffer` class, buffer management functions |
| `ui.py` | Add buffer support to `ImageWindow`, save transformed menu |
| `rois.py` | Add `get_region()`/`get_profile()` to ROI classes |
| `widgets.py` | Add progress dialog for long operations (optional) |

## Testing Strategy

1. **Unit tests for ImageBuffer**
   - Create, write, read, export
   - Verify chunk handling
   - Test cleanup

2. **Integration tests**
   - Load large .ims file, convert to buffer
   - Apply rotation, save
   - Verify output matches expected

3. **ROI extraction tests**
   - Rectangle, Circle, Line profiles
   - Edge cases (ROI at image boundary)

## Open Questions

1. **Buffer caching policy**: Should we cache converted buffers across sessions?
2. **Undo support**: Should transforms be reversible via buffer history?
3. **Multi-window buffers**: Can multiple windows share a buffer for comparison?

---

*Ready for implementation after approval.*
