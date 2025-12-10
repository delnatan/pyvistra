# CLAUDE.md - AI Assistant Guide for impy

This document provides comprehensive guidance for AI assistants working with the **impy** codebase.

## Project Overview

**impy** is an image analysis and ROI (Region of Interest) management tool designed for multi-dimensional microscopy data. It provides both a GUI application and a Python library for working with large image datasets, particularly Imaris (.ims) and TIFF files.

### Key Features
- Multi-dimensional image viewer (5D: Time, Z-depth, Channels, Y, X)
- Interactive ROI drawing and management (rectangle, circle, line, coordinate system)
- GPU-accelerated rendering via Vispy
- Lazy loading for efficient memory usage with large files
- Orthogonal 3D viewer with synchronized views
- Z-projection (max intensity projection)
- ROI-based analysis (intensity measurements, line profiles, cropping)
- Export capabilities (TIFF with ImageJ-compatible metadata)

### Technology Stack
- **UI Framework**: Qt (via qtpy for PyQt6/PyQt5 compatibility)
- **Rendering**: Vispy (OpenGL-based GPU rendering)
- **Data I/O**: h5py (Imaris files), tifffile (TIFF files)
- **Analysis**: NumPy, matplotlib
- **UI Components**: superqt, magicgui

## Critical Architecture Concepts

### 1. The 5D Data Model

**ALL data in impy is standardized to 5D format: `(T, Z, C, Y, X)`**

- **T**: Time points
- **Z**: Depth slices (z-stack)
- **C**: Channels (e.g., fluorescence channels)
- **Y**: Height (rows)
- **X**: Width (columns)

This standardization eliminates dimension confusion across different file formats. The `io.py` module handles normalization via `normalize_to_5d()`.

### 2. Proxy Pattern for Lazy Loading

The codebase uses proxy objects that behave like numpy arrays but load data on-demand:

- **`Imaris5DProxy`**: Wraps `ImarisReader` for .ims files (HDF5-based)
- **`Numpy5DProxy`**: Wraps in-memory numpy arrays
- **`TransposedProxy`**: On-the-fly transposition without data copying

**Benefits**: Work with multi-GB files without loading entire dataset into memory.

**Key Insight**: When you see `data[t, z, c, :, :]`, the proxy intercepts `__getitem__` and performs lazy loading from disk.

### 3. Vispy + Qt Integration

**Rendering Pipeline**:
1. Qt provides the UI framework (windows, widgets, controls)
2. Vispy provides GPU-accelerated OpenGL rendering
3. `scene.SceneCanvas` embeds as a Qt widget via `canvas.native`
4. `CompositeImageVisual` manages multiple image layers with additive blending

**Coordinate Systems**:
- Vispy uses OpenGL coordinates (bottom-left origin)
- Images use numpy indexing (top-left origin)
- Camera flip and transforms handle conversion
- ROI interactions require coordinate mapping via `node_transform()`

### 4. ROI System Architecture

**Design Pattern**: Distributed storage with central coordination

- Each `ImageWindow` maintains its own `rois` list
- `ROIManager` (singleton) provides UI and cross-window operations
- ROI base class (`rois.py`) defines common interface
- Concrete types: `CoordinateROI`, `RectangleROI`, `CircleROI`, `LineROI`

**Serialization**: JSON format with `{"type": "ClassName", "name": "...", "data": {...}}`

## Codebase Structure

### Core Modules (`impy/`)

```
impy/
├── __init__.py           # Version info (__version__ = "0.1.2")
├── __main__.py           # Entry point (creates QApplication + Toolbar)
├── io.py                 # Data loading/saving, 5D normalization, proxies
├── imaris_reader.py      # HDF5 reader for Bitplane Imaris files
├── ui.py                 # ImageWindow, Toolbar, dialogs, main UI logic
├── ortho.py              # OrthoViewer (3-panel orthogonal view)
├── visuals.py            # CompositeImageVisual (Vispy rendering)
├── rois.py               # ROI base class and concrete types
├── roi_manager.py        # ROIManager singleton (ROI list UI)
├── widgets.py            # HistogramWidget, ContrastDialog, MetadataDialog
├── analysis.py           # ROI-based measurements (@magicgui functions)
├── manager.py            # WindowManager singleton (global state)
└── theme.py              # DARK_THEME (Qt stylesheet)
```

### Supporting Files

```
tests/                    # Standalone test scripts
scripts/                  # Debug/reproduction scripts
pyproject.toml           # Package metadata, dependencies, entry points
requirements.txt         # Dependency list (duplicates pyproject.toml)
README.md                # User-facing documentation
```

### Module Responsibilities

#### **io.py** - Data I/O Layer
- **`load_image(path)`**: Main entry point, returns (proxy, metadata) tuple
- **`save_tiff(path, data, scale=None)`**: Export with ImageJ metadata
- **`normalize_to_5d(arr, dims=None)`**: Reshapes arrays to (T,Z,C,Y,X)
- **Proxies**: `Imaris5DProxy`, `Numpy5DProxy` (lazy loading interface)

**When to modify**: Adding new file formats, changing dimension conventions

#### **imaris_reader.py** - Imaris Parser
- **`ImarisReader(path)`**: Reads Bitplane Imaris .ims files (HDF5)
- Extracts metadata: voxel size, timestamps, channel info
- **`.read(c, t, z, res_level)`**: Lazy loading of specific slices/volumes
- Multi-resolution pyramid support

**When to modify**: Fixing Imaris format compatibility, metadata extraction bugs

#### **ui.py** - Main UI Logic
- **`ImageWindow`**: Primary viewer window (5D data, sliders, ROI interaction)
- **`Toolbar`**: Floating tool palette, file open, window management
- **`imshow(data, title, dims)`**: Programmatic viewer creation
- Event handlers: mouse (drag, click, move), keyboard (Esc, shortcuts)
- ROI drawing state machine

**When to modify**: Adding UI features, new ROI tools, keyboard shortcuts

#### **ortho.py** - 3D Orthogonal Viewer
- **`OrthoViewer`**: Three synchronized views (XY, ZY, ZX)
- Crosshair positioning with Shift+Click
- Physical scaling for correct aspect ratios
- `TransposedProxy` for on-the-fly reorientation

**When to modify**: 3D visualization features, camera synchronization

#### **visuals.py** - Rendering Layer
- **`CompositeImageVisual`**: Multi-channel overlay with additive blending
- Per-channel: color mapping, contrast (clim), gamma adjustment
- Auto-contrast heuristics
- Z-projection (max intensity)
- Caches current slice for analysis

**When to modify**: Rendering bugs, new visualization modes, performance optimization

#### **rois.py** - ROI Classes
- **`ROI`**: Base class (selection, handles, serialization, hit testing)
- **`CoordinateROI`**: Orthogonal axes with anterior/dorsal vectors, flip support
- **`RectangleROI`**: 4-corner draggable rectangle
- **`CircleROI`**: Center + radius handle
- **`LineROI`**: Two-point line

**When to modify**: Adding new ROI types, fixing interaction bugs

#### **roi_manager.py** - ROI Management UI
- **`ROIManager`**: Singleton (use `get_roi_manager()`)
- Window selector dropdown
- ROI list view (selection, deletion)
- Save/Load JSON
- Analysis menu integration

**When to modify**: ROI workflow changes, serialization format

#### **widgets.py** - Custom Qt Widgets
- **`HistogramWidget`**: Interactive contrast adjustment with log-scale
- **`ContrastDialog`**: Per-channel contrast, gamma, auto-contrast
- **`MetadataDialog`**: Table display of image metadata

**When to modify**: UI widget improvements, contrast algorithm changes

#### **analysis.py** - ROI Analysis
- **`plot_profile()`**: Line intensity profile
- **`crop_image()`**: Extract ROI region to new window
- **`measure_intensity()`**: Mean/std statistics
- Uses `@magicgui` for automatic GUI generation

**When to modify**: Adding new analysis functions

#### **manager.py** - Global State
- **`WindowManager`**: Singleton (tracks windows, active tool)
- Window registration/lookup by ID
- Active tool state (pointer, rect, circle, etc.)

**When to modify**: Adding global state, window lifecycle changes

## Development Setup

### Installation

```bash
# Clone repository
git clone <repository-url>
cd impy

# Install in editable mode (recommended for development)
pip install -e .
# or with uv
uv pip install -e .
```

### Running the Application

```bash
# Via entry point
impy

# Via module
python -m impy

# Programmatic (in Python scripts)
from impy.ui import imshow, run_app
import numpy as np
data = np.random.rand(10, 100, 100)  # (Z, Y, X)
viewer = imshow(data, dims='zyx')
run_app()
```

### Dependencies

**Core**:
- numpy (array operations)
- vispy (GPU rendering)
- qtpy + PyQt6 (UI framework)
- h5py (HDF5/Imaris files)
- tifffile (TIFF I/O)

**UI**:
- superqt (enhanced Qt widgets, e.g., QRangeSlider)
- magicgui (auto-generate UIs from functions)
- matplotlib (plotting)

## Key Conventions

### Code Style

1. **No strict linter enforced**: Code follows PEP 8 informally
2. **Indentation**: 4 spaces
3. **Imports**: Grouped (stdlib, third-party, local)
4. **Naming**:
   - Classes: `PascalCase` (e.g., `ImageWindow`, `ROIManager`)
   - Functions/methods: `snake_case` (e.g., `load_image`, `update_slice`)
   - Private methods: `_leading_underscore` (e.g., `_update_handles`)
   - Constants: `UPPER_CASE` (e.g., `DARK_THEME`)

### Git Workflow

**Commit Messages**: Concise, lowercase, imperative mood
```
✅ Good:
  - fixed io handling of simpler 2D RGB tiff
  - added dorsal/ventral flip on CoordinateROI
  - synced ROI manager with extant ImageWindow

❌ Avoid:
  - Fixed bug (too vague)
  - WIP commit (use descriptive message)
```

**Branch Strategy**: Feature branches (no strict naming convention observed)

### Testing

**Current State**: Manual testing with standalone scripts (no pytest/unittest runner)

**Test Files**:
- `tests/test_dims_normalization.py` - Dimension normalization logic
- `tests/test_analysis.py` - ROI analysis integration
- `tests/test_z_ui.py` - Z-projection UI
- `scripts/` - Debug/reproduction scripts

**Running Tests**: Execute scripts directly
```bash
python tests/test_dims_normalization.py
```

**When Adding Features**: Create corresponding test script in `tests/` or `scripts/`

## Common Development Tasks

### Adding a New ROI Type

1. **Create class in `rois.py`** inheriting from `ROI`
2. **Implement required methods**:
   - `__init__`: Create Vispy visuals, append to `self.visuals`
   - `update(start, end)`: Called during mouse drag
   - `_update_handles()`: Define handle positions
   - `hit_test(point)`: Return handle ID or 'center' or None
   - `move(delta)`: Translate entire ROI
   - `adjust(handle_id, new_pos)`: Move specific handle
   - `to_dict()`: Serialize geometry to dict
   - `_update_visuals_from_data()`: Reconstruct from loaded data
3. **Register in UI**: Add tool button in `Toolbar.__init__` (ui.py)
4. **Update drawing logic**: Modify `ImageWindow._on_mouse_press/move/release`

**Example**: See `CircleROI` in `rois.py:290`

### Adding File Format Support

1. **Create reader** (similar to `ImarisReader`)
2. **Add proxy class** in `io.py` (if lazy loading needed)
3. **Update `load_image()`**: Add file extension detection
4. **Normalize to 5D**: Ensure output is (T, Z, C, Y, X)
5. **Extract metadata**: Populate `meta` dict with scale, channels, etc.

**Metadata Structure**:
```python
meta = {
    "filename": str,
    "shape": tuple,  # (T, Z, C, Y, X)
    "scale": tuple,  # (sz, sy, sx) in microns
    "channels": list[dict],  # [{"name": "GFP", "wavelength": 488}, ...]
    "timestamps": list,  # Optional
}
```

### Adding Analysis Functions

1. **Create function in `analysis.py`**
2. **Decorate with `@magicgui`** for auto-GUI
3. **Parameters**: Accept ROI, data, metadata as needed
4. **Register in `ROIManager.create_analysis_menu()`**

**Example**:
```python
from magicgui import magicgui

@magicgui
def my_analysis(roi: ROI, data: np.ndarray):
    # Extract ROI region
    # Perform analysis
    # Display results
    pass
```

### Modifying the UI

**Layout Structure** (ImageWindow):
```
QVBoxLayout
  ├── SceneCanvas (Vispy)
  ├── QHBoxLayout (sliders)
  │   ├── Time slider
  │   ├── Z slider
  │   └── Channel slider/combo
  └── Info label
```

**Adding Widgets**: Modify `ImageWindow.__init__` in `ui.py`

**Styling**: Update `DARK_THEME` in `theme.py` (Qt CSS)

### Debugging Tips

1. **Print data shapes**: `print(f"Shape: {data.shape}")` (verify 5D format)
2. **Check proxy type**: `type(data)` → Imaris5DProxy or Numpy5DProxy
3. **Inspect metadata**: `print(meta)` after `load_image()`
4. **Vispy debugging**: Set `VISPY_DEBUG=1` environment variable
5. **Qt debugging**: Check console for Qt warnings
6. **ROI issues**: Print `roi.data` to inspect serialized geometry

## Important Gotchas

### 1. Dimension Order Matters

**Problem**: Mixing up dimension order causes incorrect rendering
**Solution**: ALWAYS use (T, Z, C, Y, X) order after `load_image()`

```python
# ✅ Correct
data, meta = load_image("file.ims")
slice_2d = data[0, 5, 1, :, :]  # t=0, z=5, c=1

# ❌ Wrong - will cause errors or weird rendering
slice_2d = data[0, 1, 5, :, :]  # Swapped Z and C
```

### 2. Proxy Slicing Returns NumPy Arrays

**Problem**: Slicing a proxy returns a concrete numpy array, not another proxy
**Solution**: Be aware of memory implications when slicing large regions

```python
data, _ = load_image("large_file.ims")  # data is Imaris5DProxy
volume = data[0, :, 0, :, :]  # Now volume is numpy array in RAM
```

### 3. Vispy Coordinate Transformations

**Problem**: Mouse coordinates don't directly map to data indices
**Solution**: Use `_map_event_to_image()` in `ImageWindow`

```python
# In ImageWindow event handler
data_coords = self._map_event_to_image(event)
if data_coords is None:
    return  # Click outside image bounds
x, y = data_coords
```

### 4. ROI Manager Synchronization

**Problem**: ROI Manager can get out of sync with ImageWindow
**Solution**: Always call `roi_manager.refresh_list()` after modifying `window.rois`

```python
# After adding/removing ROIs
self.rois.append(new_roi)
get_roi_manager().refresh_list()
```

### 5. Qt Event Loop

**Problem**: Calling `run_app()` or `QApplication.exec_()` multiple times
**Solution**: Only one event loop per application

```python
# ✅ Correct
app = QApplication.instance() or QApplication(sys.argv)
viewer = imshow(data)
app.exec_()  # Run once at end

# ❌ Wrong - will error
app.exec_()
app.exec_()  # Can't run twice
```

### 6. File Handle Leaks

**Problem**: Imaris files stay open (HDF5 file handles)
**Solution**: Close windows or manually close reader

```python
reader = ImarisReader("file.ims")
# ... use reader ...
reader.file.close()  # Explicitly close HDF5 file
```

### 7. Scale Metadata

**Problem**: Forgetting to preserve scale when saving TIFF
**Solution**: Always pass `scale` parameter to `save_tiff()`

```python
data, meta = load_image("input.ims")
crop = data[0, :, 0, :, :]
save_tiff("output.tif", crop, scale=meta['scale'])  # Preserve voxel size
```

## Testing Guidelines

### Unit Tests

**Location**: `tests/`

**Structure**: Standalone scripts with `if __name__ == "__main__"`

**Example**:
```python
# tests/test_my_feature.py
import numpy as np
from impy.io import normalize_to_5d

def test_normalize_3d():
    data = np.random.rand(10, 100, 100)  # (Z, Y, X)
    result = normalize_to_5d(data, dims='zyx')
    assert result.shape == (1, 10, 1, 100, 100)  # (T, Z, C, Y, X)
    print("✓ test_normalize_3d passed")

if __name__ == "__main__":
    test_normalize_3d()
```

**Running**: `python tests/test_my_feature.py`

### Integration Tests

**With GUI**: Mock or disable interactive elements

```python
# Mock matplotlib to avoid blocking
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from impy.ui import ImageWindow, run_app
# Test GUI components
```

### Debugging Scripts

**Location**: `scripts/`

**Purpose**: Reproduce specific bugs or test edge cases

**Example**: `scripts/reproduce_roi_error.py`

## Common Workflows for AI Assistants

### Investigating a Bug

1. **Reproduce**: Create minimal test case
2. **Locate**: Search codebase with `Grep` or `Task` agent
3. **Read**: Examine relevant files with `Read`
4. **Identify**: Trace data flow, check assumptions
5. **Fix**: Modify code with `Edit`
6. **Test**: Run affected test scripts
7. **Commit**: Descriptive commit message

### Implementing a Feature Request

1. **Plan**: Break down into subtasks
2. **Research**: Understand existing architecture
3. **Implement**: Add code in appropriate modules
4. **Integrate**: Update UI, register in managers
5. **Test**: Create test script
6. **Document**: Update README if user-facing
7. **Commit**: Clear commit message

### Refactoring

1. **Read first**: Understand current implementation thoroughly
2. **Plan**: Document changes, ensure backwards compatibility
3. **Small steps**: Incremental changes, test after each
4. **Preserve behavior**: Don't change functionality unintentionally
5. **Update tests**: Ensure tests still pass

## Key Files to Read First

When starting work on impy, read these files in order:

1. **README.md** - User-facing overview
2. **pyproject.toml** - Dependencies, project metadata
3. **impy/__main__.py** - Entry point (understand startup)
4. **impy/io.py** - Data model foundation (5D normalization)
5. **impy/ui.py** - Main UI logic (ImageWindow, Toolbar)
6. **impy/rois.py** - ROI system (base class, concrete types)

Then dive into specific modules as needed for your task.

## Architectural Patterns

### Singleton Pattern

**Used for**: WindowManager, ROIManager

```python
_roi_manager_instance = None

def get_roi_manager():
    global _roi_manager_instance
    if _roi_manager_instance is None:
        _roi_manager_instance = ROIManager()
    return _roi_manager_instance
```

**Why**: Single source of truth for global state

### Proxy Pattern

**Used for**: Lazy data loading (Imaris5DProxy, Numpy5DProxy)

```python
class Imaris5DProxy:
    def __getitem__(self, key):
        # Intercept slicing, load data on demand
        return self.reader.read(...)
```

**Why**: Memory efficiency with large files

### Factory Pattern

**Used for**: ROI deserialization

```python
def from_dict(cls, data, view):
    roi_type = data["type"]
    if roi_type == "RectangleROI":
        return RectangleROI(view)
    # ...
```

**Why**: Polymorphic object creation from JSON

### Observer Pattern

**Used for**: UI updates, ROI synchronization

**Why**: Decoupled communication between components

## Version Information

**Current Version**: 0.1.2 (see `impy/__init__.py`)

**Python Compatibility**: >=3.8

**Qt Backend**: PyQt6 (with PyQt5 fallback via qtpy)

## Additional Resources

- **Vispy Documentation**: https://vispy.org/
- **Qt for Python**: https://doc.qt.io/qtforpython/
- **Imaris File Format**: HDF5-based (reverse-engineered, no official spec)
- **ImageJ TIFF Metadata**: https://imagej.nih.gov/ij/developer/api/

## Summary for AI Assistants

**When working with impy**:

1. ✅ **DO**:
   - Always read files before modifying
   - Respect the 5D data convention (T, Z, C, Y, X)
   - Use `load_image()` for file I/O
   - Update ROIManager after modifying window.rois
   - Write concise, descriptive commit messages
   - Create test scripts for new features
   - Check for proxy types when debugging
   - Preserve metadata (especially scale) when saving

2. ❌ **DON'T**:
   - Assume dimension order without checking
   - Create new files when editing existing ones would suffice
   - Skip reading architecture docs before major changes
   - Forget to handle Qt/Vispy coordinate transformations
   - Run multiple Qt event loops
   - Leave HDF5 file handles open

3. 🔍 **INVESTIGATE**:
   - Use `Task` agent for exploratory searches
   - Read relevant modules before implementing features
   - Trace data flow through proxies and visualizations
   - Check recent git history for context on similar changes

**This codebase is well-structured and maintainable. Respect the existing architecture, follow conventions, and you'll have a smooth development experience.**

---

*Last Updated: 2025-12-10*
*Version: 0.1.2*
