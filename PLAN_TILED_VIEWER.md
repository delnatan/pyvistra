# Tiled Image Viewer Implementation Plan

## Overview

A gallery-style tiled image viewer for comparing multiple images, inspired by single-particle cryo-EM software (e.g., RELION, cryoSPARC particle galleries). Designed for fluorescently-labeled neuron crops, but generic enough to handle any collection of 2D images (e.g., sub-ROI crops, particle picks).

## Design Goals

1. **Memory-efficient**: Capped tiles per page with lazy loading
2. **Responsive**: GPU-accelerated rendering via Vispy (one SceneCanvas per tile)
3. **Informative**: Quick access to per-image metadata
4. **Flexible contrast**: Per-tile contrast adjustment for wide dynamic range
5. **Ergonomic input**: Drag-drop multiple files or folder to trigger
6. **Fluid layout**: Tiles flow/wrap naturally, resize with window

---

## Architecture

### New Files to Create

```
impy/
├── tiled_viewer.py    # TiledViewer window + TileWidget component
└── (modify existing)
    └── ui.py          # Update Toolbar drop handling
```

### Class Hierarchy

```
TiledViewer (QMainWindow)
├── Toolbar (page nav, tile size slider, per-page selector)
├── QScrollArea
│   └── FlowContainer (custom widget with flow layout)
│       └── TileWidget[] (one per loaded image)
│           ├── SceneCanvas (Vispy) for rendering
│           ├── CompositeImageVisual (reused from visuals.py)
│           └── Info label (filename, dimensions)
└── Status bar (page info, tile count)
```

---

## Key Design Decisions

### Flow Layout with Scrolling

- Tiles flow left-to-right, wrapping to next row when window width exceeded
- Vertical scrollbar for viewing all tiles on current page
- Window resize → tiles reflow naturally (like CSS flexbox wrap)
- No fixed grid — number of columns depends on tile size and window width

### Tile Size vs Zoom (Independent)

| Concept | What it controls | User control |
|---------|------------------|--------------|
| **Tile size** | Widget dimensions in layout (px) | Slider: 100–400px |
| **Zoom** | Pan/zoom within Vispy canvas | Mouse wheel/drag inside tile |

Changing tile size reflows the layout. Zooming inside a tile does not affect layout.

### Pagination for Memory Management

- **Max tiles per page**: Capped (e.g., 50–100) to limit GPU memory
- **Per-page selector**: User chooses how many images to load at once
- **Page navigation**: Load/unload batches of images
- Future: HDF5-backed tile storage for dynamic fetching

### Input: List of File Paths

Generic interface — accepts any list of image paths:
- Drag-drop multiple files → TiledViewer
- Drag-drop folder → collect all images recursively
- Programmatic: `TiledViewer(image_paths=[...])`

---

## Component Design

### 1. TiledViewer (Main Window)

```python
class TiledViewer(QMainWindow):
    """Gallery view for multiple images with flow layout."""

    def __init__(self, image_paths: list[str], tiles_per_page: int = 50):
        self.image_paths = image_paths
        self.tiles_per_page = tiles_per_page  # Max tiles loaded at once
        self.current_page = 0
        self.tile_size = 200  # Default tile size in pixels

        self.tile_widgets: list[TileWidget] = []

        self._setup_ui()
        self._load_current_page()
```

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│ ◀ Page 1/3 ▶ │ Tiles/page: [50▼] │ Size: [────●────] │ [Auto All]│
├─────────────────────────────────────────────────────────────────┤
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       ▲│
│ │     │ │     │ │     │ │     │ │     │ │     │ │     │       ░│
│ │ IMG │ │ IMG │ │ IMG │ │ IMG │ │ IMG │ │ IMG │ │ IMG │       ░│
│ │     │ │     │ │     │ │     │ │     │ │     │ │     │       ░│
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       ░│
│ name_01 name_02 name_03 name_04 name_05 name_06 name_07       ░│
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ...                   ░│
│ │     │ │     │ │     │ │     │ │     │                       ░│
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                       ▼│
├─────────────────────────────────────────────────────────────────┤
│ Showing 1-50 of 127 images                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2. FlowLayout (Custom Layout Manager)

Qt doesn't have a built-in flow layout, so we implement one:

```python
class FlowLayout(QLayout):
    """Layout that arranges widgets in a flowing left-to-right, top-to-bottom manner."""

    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        # Calculate based on items
        ...

    def doLayout(self, rect, test_only=False):
        """Arrange items in flow layout within given rect."""
        x, y = rect.x(), rect.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            space_x = self._spacing
            space_y = self._spacing

            next_x = x + widget.sizeHint().width() + space_x

            if next_x - space_x > rect.right() and line_height > 0:
                # Wrap to next row
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + widget.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                widget.setGeometry(QRect(QPoint(x, y), widget.sizeHint()))

            x = next_x
            line_height = max(line_height, widget.sizeHint().height())

        return y + line_height - rect.y()
```

### 3. TileWidget (Individual Tile)

```python
class TileWidget(QFrame):
    """Single tile displaying one image with Vispy canvas."""

    def __init__(self, tile_size: int = 200):
        super().__init__()
        self._tile_size = tile_size

        # Vispy canvas for GPU rendering
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="#1a1a1a")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1

        # Renderer (reuse existing CompositeImageVisual)
        self.renderer = CompositeImageVisual(self.view)

        # Data
        self.data = None   # 5D array/proxy
        self.meta = None   # Metadata dict
        self.file_path = None

        # UI
        self.info_label = QLabel()  # Shows filename
        self._setup_ui()

    def set_tile_size(self, size: int):
        """Update tile size (does not affect internal zoom)."""
        self._tile_size = size
        self.setFixedSize(size, size + 20)  # +20 for info label
        self.canvas.native.setFixedSize(size, size)
        self.updateGeometry()

    def load(self, path: str):
        """Load image from path."""
        self.file_path = path
        self.data, self.meta = load_image(path)

        # Display first frame/slice
        self.renderer.set_data(self.data)
        self.renderer.update_slice(0, 0)  # t=0, z=middle
        self._fit_view()
        self._update_info()

    def unload(self):
        """Release memory."""
        self.renderer.clear()
        self.data = None
        self.meta = None

    def _fit_view(self):
        """Reset camera to fit image."""
        if self.data is not None:
            h, w = self.data.shape[-2:]
            self.view.camera.set_range(x=(0, w), y=(0, h), margin=0)
```

**Tile Layout:**
```
┌─────────────────┐
│                 │
│  Vispy Canvas   │  ← Fixed size (tile_size × tile_size)
│  (pan/zoom OK)  │
│                 │
├─────────────────┤
│ filename.tif    │  ← Info label (truncated, tooltip for full)
└─────────────────┘
```

### 4. Per-Tile Contrast (Context Menu)

Right-click on tile opens context menu:

```python
def _show_context_menu(self, pos):
    menu = QMenu(self)

    # Contrast action
    contrast_action = menu.addAction("Adjust Contrast...")
    contrast_action.triggered.connect(self._show_contrast_dialog)

    # Auto contrast
    auto_action = menu.addAction("Auto Contrast")
    auto_action.triggered.connect(self._auto_contrast)

    menu.addSeparator()

    # Open in new window
    open_action = menu.addAction("Open in Viewer")
    open_action.triggered.connect(self._open_in_viewer)

    # Show metadata
    meta_action = menu.addAction("Show Metadata")
    meta_action.triggered.connect(self._show_metadata)

    menu.exec_(self.mapToGlobal(pos))
```

**Contrast Dialog**: Reuse existing `ContrastDialog` from widgets.py, or create a simplified popup version.

---

## Workflow

### Drag-and-Drop Trigger (Modified Toolbar)

```python
# In ui.py Toolbar class

def dropEvent(self, event: QDropEvent):
    files = [u.toLocalFile() for u in event.mimeData().urls()]

    # Collect supported image files
    supported_ext = {'.ims', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    image_files = []

    for f in files:
        if os.path.isdir(f):
            # Folder: collect all images recursively
            for root, _, names in os.walk(f):
                for name in names:
                    if Path(name).suffix.lower() in supported_ext:
                        image_files.append(os.path.join(root, name))
        elif Path(f).suffix.lower() in supported_ext:
            image_files.append(f)

    # Sort by filename
    image_files.sort()

    if len(image_files) > 1:
        # Multiple files → TiledViewer
        from .tiled_viewer import TiledViewer
        viewer = TiledViewer(image_files)
        viewer.show()
    elif len(image_files) == 1:
        # Single file → regular ImageWindow
        self.spawn_viewer(image_files[0])
```

### Page Loading/Unloading

```python
def _load_current_page(self):
    """Load tiles for current page."""
    # Clear existing tiles
    self._clear_tiles()

    # Calculate slice
    start = self.current_page * self.tiles_per_page
    end = min(start + self.tiles_per_page, len(self.image_paths))

    # Create and load tiles
    for path in self.image_paths[start:end]:
        tile = TileWidget(self.tile_size)
        tile.load(path)
        tile.setContextMenuPolicy(Qt.CustomContextMenu)
        tile.customContextMenuRequested.connect(
            lambda pos, t=tile: self._show_tile_context_menu(t, pos)
        )
        self.flow_layout.addWidget(tile)
        self.tile_widgets.append(tile)

    self._update_status()

def _clear_tiles(self):
    """Remove and unload all current tiles."""
    for tile in self.tile_widgets:
        tile.unload()
        self.flow_layout.removeWidget(tile)
        tile.deleteLater()
    self.tile_widgets.clear()
```

### Tile Size Adjustment

```python
def _on_tile_size_changed(self, value: int):
    """Handle tile size slider change."""
    self.tile_size = value
    for tile in self.tile_widgets:
        tile.set_tile_size(value)

    # Trigger reflow
    self.flow_container.updateGeometry()
    self.scroll_area.widget().adjustSize()
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` or `PgUp` / `PgDn` | Previous / Next page |
| `Home` / `End` | First / Last page |
| `A` | Auto-contrast all visible tiles |
| `+` / `-` | Increase / decrease tile size |
| `Ctrl+O` | Open files dialog |

---

## Memory Considerations

### Tile Limits

| Tiles/page | Est. GPU memory* | Recommended for |
|------------|------------------|-----------------|
| 25         | ~100 MB          | Large images (1K+) |
| 50         | ~200 MB          | Medium images (512px) |
| 100        | ~400 MB          | Small images (<256px) |

*Assuming 512×512×3ch×float32 per tile + texture overhead

### Optimization Strategies

1. **Unload on page change**: Call `tile.unload()` to release GPU textures
2. **Float32 textures**: Use `np.float32` (not float64) for GPU data
3. **Single slice**: Only load t=0, z=middle by default
4. **Future**: Virtual scrolling (create/destroy tiles as they scroll into view)

---

## Future Extensions

1. **HDF5 tile storage**: Store many small images in single HDF5 file for efficient I/O
2. **Virtual scrolling**: Only render visible tiles for very large collections
3. **Tile selection**: Multi-select for batch operations (export, delete)
4. **Sorting**: Sort by name, date, or custom metadata field
5. **Filtering**: Show only tiles matching criteria
6. **Linked zoom**: Optional synchronized pan/zoom across all tiles

---

## Implementation Phases

### Phase 1: MVP
1. Create `tiled_viewer.py` with `TiledViewer` and `TileWidget`
2. Implement `FlowLayout` for tile arrangement
3. Basic Vispy rendering per tile (reuse `CompositeImageVisual`)
4. Pagination (load/unload pages)
5. Tile size slider

### Phase 2: Contrast & Info
6. Per-tile context menu (right-click)
7. Contrast adjustment (reuse `ContrastDialog`)
8. Auto-contrast all button
9. Info labels with filename

### Phase 3: Integration
10. Modify `Toolbar.dropEvent()` for multi-file trigger
11. "Open in Viewer" action (spawn `ImageWindow`)
12. Keyboard shortcuts

### Phase 4: Polish
13. Status bar with page/count info
14. Tooltips with full path and metadata
15. Remember settings (QSettings)
16. Error handling for failed loads
