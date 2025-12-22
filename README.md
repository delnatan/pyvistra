# pyvistra

Image analysis and ROI management tool.

## Installation

To install the package for local development (editable mode), use the `-e` (editable) flag. This allows you to modify the code and see changes immediately without reinstalling.

### Using pip

```bash
pip install -e .
```

### Using uv

```bash
uv pip install -e .
```

## Usage

Run the application:

```bash
pyvistra
# or
python -m pyvistra
```

## Library Usage

You can use `pyvistra` as a library to read and process images programmatically.

### Reading Imaris Files

You can use the `ImarisReader` class to read `.ims` files directly:

```python
from pyvistra.imaris_reader import ImarisReader

# Open the file
reader = ImarisReader('path/to/file.ims')

# Print metadata
print(f"Shape: {reader.shape}")  # (Time, Channels, Z, Y, X)
print(f"Channels: {reader.channels_info}")
print(f"Voxel Size: {reader.voxel_size}")

# Read specific data
# Read timepoint 0, channel 1, z-slice 5
plane = reader.read(c=1, t=0, z=5)

# Read full volume for timepoint 0, channel 0
volume = reader.read(c=0, t=0, z=None)
```

### Loading a 3D Timelapse

To easily load a 3D timelapse, you can use the `load_image` function. The returned proxy object behaves like a 5D numpy array.

```python
import numpy as np
from pyvistra.io import load_image

# Load image using the high-level loader
# Returns a proxy object that behaves like a numpy array
data, meta = load_image('path/to/file.ims')

# The proxy shape is standardized to (Time, Z, Channel, Y, X)
print(f"Data shape: {data.shape}")

# Get dimensions
n_timepoints = data.shape[0]
n_channels = data.shape[2]
height = data.shape[3]
width = data.shape[4]

# Read specific data
# Read timepoint 0, full Z-stack, channel 0
volume = data[0, :, 0]
print(f"Volume shape: {volume.shape}") # (Z, Y, X)
```

### Saving Images

You can save 5D arrays (or crops) to TIFF format using `save_tiff`. This preserves voxel size and dimension order metadata compatible with ImageJ.

```python
from pyvistra.io import save_tiff

# Save the volume we just read
# meta['scale'] contains the voxel size (z, y, x)
save_tiff("output.tif", volume, scale=meta['scale'])
```

### Interactive IPython Usage

`pyvistra` can be used interactively in IPython or Jupyter notebooks. First, enable Qt event loop integration:

```python
# In IPython, enable Qt event loop integration
%gui qt

# Now you can use imshow() interactively
import numpy as np
from pyvistra.ui import imshow

# Create some test data
data = np.random.rand(20, 256, 256)  # (Z, Y, X)

# Display image - returns immediately, window is interactive
viewer = imshow(data, title="Random Data", dims="zyx")

# You can continue working in IPython while the viewer is open
# Access the viewer's data
print(viewer.img_data.shape)  # (1, 20, 1, 256, 256) - normalized to 5D

# Load and display a file
from pyvistra.io import load_image
data, meta = load_image('path/to/file.ims')
viewer2 = imshow(data[0, :, 0], title="Channel 0", dims="zyx")
```

If you're running from a regular Python script (not IPython), you need to start the Qt event loop manually:

```python
from pyvistra.ui import imshow, run_app
import numpy as np

data = np.random.rand(10, 100, 100)
viewer = imshow(data, dims="zyx")

# Start the event loop (blocks until all windows are closed)
run_app()
```
