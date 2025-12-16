# Tiled Image Viewer Implementation Plan

## Overview

A gallery-style tiled image viewer for comparing multiple fluorescently-labeled neuron images, inspired by single-particle cryo-EM software (e.g., RELION, cryoSPARC particle galleries).

## Design Goals

1. **Memory-efficient**: User-specified "images per page" with lazy loading
2. **Responsive**: GPU-accelerated rendering via Vispy (one SceneCanvas per tile)
3. **Informative**: Quick access to per-image metadata
4. **Flexible contrast**: Per-tile contrast adjustment for wide dynamic range
5. **Ergonomic input**: Drag-drop multiple files or folder to trigger

---

## Architecture

### New Files to Create

```
impy/
├── tiled_viewer.py    # TiledViewer window + TileWidget component
└── (modify existing)
    ├── ui.py          # Update Toolbar drop handling
    └── io.py          # Add batch loading utility
```

### Class Hierarchy

```
TiledViewer (QMainWindow)
├── Navigation bar (page controls, per-page count selector)
├── QScrollArea containing QGridLayout
│   └── TileWidget[] (one per visible image)
│       ├── SceneCanvas (Vispy) for rendering
│       ├── CompositeImageVisual (reused from visuals.py)
│       ├── Info overlay (filename, dimensions, scale)
│       └── Mini contrast control (click to expand)
└── Status bar (current page, total images)
```

---

## Component Design

### 1. TiledViewer (Main Window)

```python
class TiledViewer(QMainWindow):
    """Gallery view for multiple images in a paginated grid."""

    def __init__(self, image_paths: list[str], images_per_page: int = 9):
        # Parameters
        self.image_paths = image_paths      # All file paths
        self.images_per_page = images_per_page
        self.current_page = 0

        # Loaded data (only current page)
        self.tile_widgets: list[TileWidget] = []

        # UI Layout
        self._setup_ui()
        self._load_current_page()
```

**Key features:**
- Pagination with Previous/Next buttons + page indicator
- "Images per page" dropdown: 4, 9, 16, 25 (2x2, 3x3, 4x4, 5x5)
- Keyboard navigation: Left/Right arrows for pages
- "Open in separate window" action per tile (spawns regular ImageWindow)
- Global zoom level sync (optional checkbox)

### 2. TileWidget (Individual Tile)

```python
class TileWidget(QWidget):
    """Single tile in the gallery grid."""

    def __init__(self, parent: TiledViewer):
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1

        self.renderer = CompositeImageVisual(self.view)
        self.data = None       # 5D proxy/array
        self.meta = None       # Metadata dict
        self.selected = False  # For multi-select operations

        # Info overlay (semi-transparent label)
        self.info_label = QLabel()  # Shows filename, dimensions

        # Mini contrast button (click to show popup)
        self.contrast_btn = QToolButton()
```

**Layout per tile:**
```
┌─────────────────────────────┐
│ [filename.tif]    [⚙️] [🔍] │  <- Header: name, contrast btn, open btn
├─────────────────────────────┤
│                             │
│     Vispy SceneCanvas       │  <- GPU-rendered image
│     (CompositeImageVisual)  │
│                             │
├─────────────────────────────┤
│ 512x512 | 3 ch | 0.1µm/px   │  <- Footer: quick metadata
└─────────────────────────────┘
```

### 3. Contrast Popup (Per-Tile)

A lightweight popup anchored to the tile for quick contrast adjustment:

```python
class TileContrastPopup(QWidget):
    """Compact contrast controls for a single tile."""

    def __init__(self, tile: TileWidget):
        # Mini histogram (smaller than ContrastDialog)
        self.histogram = HistogramWidget()

        # Channel selector (if multi-channel)
        self.channel_combo = QComboBox()

        # Auto-contrast button
        self.auto_btn = QPushButton("Auto")

        # Gamma slider (compact)
        self.gamma_slider = QSlider(Qt.Horizontal)
```

**Interaction:**
- Click contrast button → popup appears below/beside tile
- Click elsewhere → popup closes
- "Apply to all" button copies settings to all visible tiles

---

## Workflow

### Drag-and-Drop Trigger

**Modified `Toolbar.dropEvent()` in ui.py:**

```python
def dropEvent(self, event: QDropEvent):
    files = [u.toLocalFile() for u in event.mimeData().urls()]

    # Filter to supported image files
    supported = ['.ims', '.tif', '.tiff', '.png', '.jpg']
    image_files = []

    for f in files:
        if os.path.isdir(f):
            # Folder dropped: collect all images inside
            for root, _, names in os.walk(f):
                for name in names:
                    if any(name.lower().endswith(ext) for ext in supported):
                        image_files.append(os.path.join(root, name))
        elif any(f.lower().endswith(ext) for ext in supported):
            image_files.append(f)

    if len(image_files) > 1:
        # Multiple files → open TiledViewer
        self._open_tiled_viewer(image_files)
    elif len(image_files) == 1:
        # Single file → open regular ImageWindow
        self.spawn_viewer(image_files[0])
```

### Page Loading

```python
def _load_current_page(self):
    """Load only images for current page (memory efficiency)."""
    start_idx = self.current_page * self.images_per_page
    end_idx = min(start_idx + self.images_per_page, len(self.image_paths))

    # Clear previous tiles
    for tile in self.tile_widgets:
        tile.unload()  # Release memory
    self.tile_widgets.clear()

    # Load new tiles
    for i, path in enumerate(self.image_paths[start_idx:end_idx]):
        tile = TileWidget(self)
        tile.load(path)  # Calls load_image() from io.py

        row, col = divmod(i, self.grid_cols)
        self.grid_layout.addWidget(tile, row, col)
        self.tile_widgets.append(tile)
```

### Synchronized Zoom (Optional)

```python
def _link_cameras(self, enabled: bool):
    """Link all tile cameras for synchronized pan/zoom."""
    if not self.tile_widgets:
        return

    primary = self.tile_widgets[0].view.camera
    for tile in self.tile_widgets[1:]:
        if enabled:
            tile.view.camera.link(primary)
        else:
            tile.view.camera.link(None)  # Unlink
```

---

## Metadata Display

### Info Overlay

Each tile shows:
- **Filename** (truncated if long, full on hover tooltip)
- **Dimensions**: e.g., "512×512 | Z:10 | C:3"
- **Scale**: e.g., "0.108 µm/px" (if available)
- **Dynamic range**: e.g., "16-bit | 234–4095"

### Detailed Metadata Dialog

Right-click tile → "Show Metadata" → Opens `MetadataDialog` (existing widget)

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / Next page |
| `Home` / `End` | First / Last page |
| `+` / `-` | Zoom in/out all tiles (if linked) |
| `A` | Auto-contrast all visible tiles |
| `Enter` | Open selected tile in new ImageWindow |
| `Esc` | Close contrast popup |

---

## UI Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│ Tiled Viewer - 27 images                             [_][□][×] │
├─────────────────────────────────────────────────────────────────┤
│ ◀ Page 1 of 3 ▶    Per page: [9 ▼]    [🔗 Link zoom]  [Auto All]│
├─────────────────────────────────────────────────────────────────┤
│ ┌───────────┐  ┌───────────┐  ┌───────────┐                    │
│ │neuron_01  │  │neuron_02  │  │neuron_03  │                    │
│ │  ┌─────┐  │  │  ┌─────┐  │  │  ┌─────┐  │                    │
│ │  │     │  │  │  │     │  │  │  │     │  │                    │
│ │  │ IMG │  │  │  │ IMG │  │  │  │ IMG │  │                    │
│ │  │     │  │  │  └─────┘  │  │  │     │  │                    │
│ │  └─────┘  │  │256×256|1ch│  │  └─────┘  │                    │
│ │512×512|3ch│  └───────────┘  │512×512|3ch│                    │
│ └───────────┘                  └───────────┘                    │
│ ┌───────────┐  ┌───────────┐  ┌───────────┐                    │
│ │neuron_04  │  │neuron_05  │  │neuron_06  │                    │
│ │  ┌─────┐  │  │  ┌─────┐  │  │  ┌─────┐  │                    │
│ │  │     │  │  │  │     │  │  │  │     │  │                    │
│ │  │ IMG │  │  │  │ IMG │  │  │  │ IMG │  │                    │
│ │  │     │  │  │  └─────┘  │  │  │     │  │                    │
│ │  └─────┘  │  │256×256|2ch│  │  └─────┘  │                    │
│ │512×512|3ch│  └───────────┘  │512×512|3ch│                    │
│ └───────────┘                  └───────────┘                    │
│ ┌───────────┐  ┌───────────┐  ┌───────────┐                    │
│ │neuron_07  │  │neuron_08  │  │neuron_09  │                    │
│ │  ...      │  │  ...      │  │  ...      │                    │
│ └───────────┘  └───────────┘  └───────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│ Showing 1-9 of 27 images                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Phase 1: Core TiledViewer (MVP)
1. Create `tiled_viewer.py` with `TiledViewer` class
2. Implement `TileWidget` with basic Vispy rendering
3. Add pagination logic (load/unload pages)
4. Integrate with `Toolbar.dropEvent()` for multi-file drops

### Phase 2: Contrast Controls
5. Add per-tile contrast popup (`TileContrastPopup`)
6. Implement "Auto Contrast All" button
7. Add "Apply to All" functionality

### Phase 3: Metadata & Info
8. Add info overlay to each tile (filename, dimensions)
9. Right-click context menu with "Show Metadata", "Open in Window"
10. Tooltip with full path on hover

### Phase 4: Polish
11. Keyboard navigation
12. Camera linking (synchronized zoom)
13. Folder drop support
14. Remember per-page setting (QSettings)

---

## Memory Considerations

| Images | Per-page | Loaded tiles | Approx. memory* |
|--------|----------|--------------|-----------------|
| 100    | 9        | 9            | ~150 MB         |
| 100    | 16       | 16           | ~260 MB         |
| 100    | 25       | 25           | ~400 MB         |

*Assuming 512×512×3ch×float32 ≈ 3 MB per image + GPU texture overhead

**Optimization strategies:**
- Use `dtype=np.float32` for GPU textures (not float64)
- Unload GPU textures when changing pages (`renderer.clear()`)
- Consider thumbnail mode for very large images (load at reduced resolution)

---

## Open Questions for User

1. **Grid layout**: Fixed square grid (3×3, 4×4) or flexible rows based on aspect ratio?
2. **Selection**: Should tiles be selectable for batch operations (delete, export, etc.)?
3. **Sorting**: Sort by filename, date, or manual reordering?
4. **Thumbnail mode**: For very large images (>2K), show thumbnails first, full res on click?
5. **Channel display**: In composite mode only, or allow single-channel view per tile?

---

## Alternative Considered: Single Large Canvas

Instead of multiple SceneCanvas instances, one large canvas with a grid of Image visuals:

**Pros**: Single GPU context, potentially more efficient
**Cons**: Complex clipping, harder per-tile interaction, camera linking trickier

**Decision**: Multiple canvases (like OrthoViewer) - simpler, proven pattern, independent pan/zoom per tile.
