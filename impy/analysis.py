import numpy as np
import matplotlib.pyplot as plt
from magicgui import magicgui
from .rois import LineROI, RectangleROI, CircleROI

@magicgui(call_button="Plot Profile")
def plot_profile(image_data, roi: LineROI):
    """
    Plot intensity profile along the line.
    """
    if not isinstance(roi, LineROI):
        print("Error: ROI must be a LineROI")
        return

    # Get points
    p1 = roi.data.get("p1")
    p2 = roi.data.get("p2")
    if not p1 or not p2:
        return

    # Extract profile
    # Simple interpolation
    num_points = int(np.linalg.norm(np.array(p2) - np.array(p1)))
    x, y = np.linspace(p1[0], p2[0], num_points), np.linspace(p1[1], p2[1], num_points)
    
    # Extract from current T/Z/C? 
    # image_data is (T, Z, C, Y, X).
    # We need to know which slice is active. 
    # For now, let's assume we pass the *sliced* data (2D) or handle it.
    # But the signature asks for `image_data`.
    # Let's assume we pass the full 5D array and maybe extra args for T/Z/C?
    # Or better: The caller (ROIManager) extracts the current 2D plane and passes it.
    
    # Let's assume image_data is 2D (Y, X) for simplicity in this prototype.
    if image_data.ndim != 2:
        print("Error: Expected 2D image data for profile")
        return

    # Map coordinates to integers
    xi = x.astype(int)
    yi = y.astype(int)
    
    # Clip
    h, w = image_data.shape
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi = xi[valid]
    yi = yi[valid]
    
    profile = image_data[yi, xi]
    
    plt.figure()
    plt.plot(profile)
    plt.title(f"Profile: {roi.name}")
    plt.xlabel("Distance (px)")
    plt.ylabel("Intensity")
    plt.show()

@magicgui(call_button="Crop Image")
def crop_image(image_data, roi: RectangleROI):
    """
    Crop the image to the ROI bounds and show in new window.
    """
    from .ui import imshow
    if not isinstance(roi, RectangleROI):
        print("Error: ROI must be a RectangleROI")
        return
        
    p1 = roi.data.get("p1")
    p2 = roi.data.get("p2")
    if not p1 or not p2:
        return
        
    x1, y1 = p1
    x2, y2 = p2
    l, r = int(min(x1, x2)), int(max(x1, x2))
    t, b = int(min(y1, y2)), int(max(y1, y2))
    
    # Handle 2D vs 5D
    # If 5D, crop Y and X dims
    if image_data.ndim == 5:
        # (T, Z, C, Y, X)
        cropped = image_data[:, :, :, t:b, l:r]
    elif image_data.ndim == 2:
        cropped = image_data[t:b, l:r]
    else:
        print(f"Unsupported dims: {image_data.ndim}")
        return
        
    imshow(cropped, title=f"Crop: {roi.name}")

@magicgui(call_button="Measure")
def measure_intensity(image_data, roi):
    """
    Measure mean/std intensity within ROI bounding box.
    """
    # Get bounds
    if isinstance(roi, RectangleROI):
        p1 = roi.data.get("p1")
        p2 = roi.data.get("p2")
        l, r = int(min(p1[0], p2[0])), int(max(p1[0], p2[0]))
        t, b = int(min(p1[1], p2[1])), int(max(p1[1], p2[1]))
    elif isinstance(roi, CircleROI):
        c = roi.data.get("center")
        r_val = roi.circle.radius
        l, r = int(c[0] - r_val), int(c[0] + r_val)
        t, b = int(c[1] - r_val), int(c[1] + r_val)
    else:
        print("Unsupported ROI for measure")
        return

    if image_data.ndim == 2:
        # Clip
        h, w = image_data.shape
        l = max(0, l); r = min(w, r)
        t = max(0, t); b = min(h, b)
        
        region = image_data[t:b, l:r]
        mn = np.mean(region)
        std = np.std(region)
        print(f"ROI {roi.name}: Mean={mn:.2f}, Std={std:.2f}")
    else:
        print("Measurement currently supports 2D slice only")
