# impy

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
impy
# or
python -m impy
```

## Library Usage

You can use `impy` as a library to read and process images programmatically.

### Reading Imaris Files

You can use the `ImarisReader` class to read `.ims` files directly:

```python
from impy.imaris_reader import ImarisReader

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

### Loading a 3D Timelapse in Max Projection

To easily load a 3D timelapse as a max projection along the Z-axis, you can use the `load_image` function. The returned proxy object supports on-the-fly max projection when you slice the Z-dimension.

```python
import numpy as np
from impy.io import load_image

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

# Pre-allocate array for the projected timelapse (Time, Channel, Y, X)
# We remove the Z dimension since we are projecting it
timelapse_max_proj = np.zeros((n_timepoints, n_channels, height, width), dtype=data.dtype)

# Loop through timepoints and channels
for t in range(n_timepoints):
    for c in range(n_channels):
        # Slicing the Z-dimension (2nd axis) with ':' triggers the max-projection
        # data[t, :, c] returns the max projection of the Z-stack for that time and channel
        timelapse_max_proj[t, c] = data[t, :, c]

print(f"Projected shape: {timelapse_max_proj.shape}")
```
