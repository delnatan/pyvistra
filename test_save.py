import numpy as np
import tifffile
from impy.io import save_tiff

# Create dummy data (T, Z, C, Y, X)
data = np.random.randint(0, 255, (1, 5, 2, 64, 64), dtype=np.uint8)
scale = (0.5, 0.1, 0.1) # Z, Y, X

save_tiff("test_output.tif", data, scale=scale)

# Read back and check metadata
with tifffile.TiffFile("test_output.tif") as tif:
    print("Axes:", tif.series[0].axes)
    print("Shape:", tif.series[0].shape)
    
    # Check resolution (tags)
    # ImageJ metadata is in tif.imagej_metadata
    print("ImageJ Metadata:", tif.imagej_metadata)
    
    # Resolution tags are usually XResolution, YResolution
    p = tif.pages[0]
    print("XRes:", p.tags["XResolution"].value)
    print("YRes:", p.tags["YResolution"].value)
