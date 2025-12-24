"""
Lab module for prototyping analysis pipelines.

This file is for experimentation. Edit freely and use reload('lab')
in the console to pick up changes without restarting.

Once a function is vetted, move it to analysis.py.
"""

import numpy as np
from .rois import RectangleROI, LineROI


def extract_lane_profiles(image_2d, rois, axis='y'):
    """
    Extract 1D intensity profiles from RectangleROI lanes.

    Args:
        image_2d: 2D numpy array (Y, X)
        rois: List of ROIs (filters to RectangleROI only)
        axis: 'y' to average across width (default), 'x' to average across height

    Returns:
        list of (name, profile) tuples
    """
    lanes = [r for r in rois if isinstance(r, RectangleROI)]
    profiles = []

    for lane in lanes:
        p1, p2 = lane.data['p1'], lane.data['p2']
        x1, x2 = int(min(p1[0], p2[0])), int(max(p1[0], p2[0]))
        y1, y2 = int(min(p1[1], p2[1])), int(max(p1[1], p2[1]))

        # Clip to image bounds
        h, w = image_2d.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        region = image_2d[y1:y2, x1:x2]

        if axis == 'y':
            # Average across width (X) to get profile along Y
            profile = np.mean(region, axis=1)
        else:
            # Average across height (Y) to get profile along X
            profile = np.mean(region, axis=0)

        profiles.append((lane.name, profile))

    return profiles


def plot_lanes(image_2d, rois, **kwargs):
    """
    Extract and plot lane profiles.

    Args:
        image_2d: 2D numpy array
        rois: List of ROIs
        **kwargs: Passed to extract_lane_profiles

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    profiles = extract_lane_profiles(image_2d, rois, **kwargs)

    if not profiles:
        print("No RectangleROI lanes found")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, profile in profiles:
        ax.plot(profile, label=name)

    ax.set_xlabel('Distance (px)')
    ax.set_ylabel('Intensity')
    ax.legend()
    plt.show()
    return fig
