import os

import numpy as np
import tifffile

from .imaris_reader import ImarisReader


class Imaris5DProxy:
    """
    Wraps ImarisReader to behave like a 5D numpy array (Time, Z, Channel, Y, X).
    This allows Vispy to 'slice' it without loading the whole file.
    """

    def __init__(self, reader):
        self.reader = reader
        # ImarisReader shape is (T, C, Z, Y, X)
        # We want (T, Z, C, Y, X) to match our application standard
        t, c, z, y, x = reader.shape
        self.shape = (t, z, c, y, x)
        self.dtype = reader.dtype
        self.ndim = 5

    def __getitem__(self, key):
        """
        Intercepts slicing: data[t, z, c, y, x]
        Vispy usually requests: data[t_idx, z_idx, :, :, :]
        """
        # Ensure key is a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Fill missing dimensions with full slices
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))

        t_idx, z_idx, c_idx, y_idx, x_idx = key

        # Resolve T and Z to integers (Visuals only request single planes)
        t = t_idx if isinstance(t_idx, int) else 0
        # z = z_idx if isinstance(z_idx, int) else 0  <-- OLD BUGGY LINE
        # We now support slices for Z
        z = z_idx

    # --- Handle Channel Slicing ---
        # If c_idx is a slice (e.g. :), we must read multiple channels and stack them.
        if isinstance(c_idx, slice):
            # Calculate range of channels requested
            start, stop, step = c_idx.indices(self.shape[2])
            channels = range(start, stop, step)

            planes = []
            for c in channels:
                planes.append(self._read_plane(c, t, z))

            # Stack into (C, Y, X)
            stack = np.array(planes)

            # Apply Y and X slicing to the stack
            # Note: stack is (C, Y, X), so we apply y_idx to dim 1, x_idx to dim 2
            return stack[:, y_idx, x_idx]

        else:
            # Single channel requested
            c = c_idx
            data_plane = self._read_plane(c, t, z)
            return data_plane[y_idx, x_idx]

    def _read_plane(self, c, t, z):
        """
        Helper to read a single plane or a Z-projection.
        z can be an int or a slice.
        """
        if isinstance(z, slice):
            # Z-Projection (Max Intensity)
            start, stop, step = z.indices(self.shape[1])
            z_indices = range(start, stop, step)
            
            if len(z_indices) == 0:
                # Return empty or zeros? Let's return zeros of correct shape
                return np.zeros((self.shape[3], self.shape[4]), dtype=self.dtype)

            # Read all planes in range
            stack = []
            for z_i in z_indices:
                stack.append(self.reader.read(c=c, t=t, z=z_i))
            
            # Max Projection
            return np.max(stack, axis=0)
        else:
            return self.reader.read(c=c, t=t, z=z)


class Numpy5DProxy:
    """
    Wraps a 5D numpy array (T, Z, C, Y, X) to support Z-projection slicing.
    """
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndim = 5

    def __getitem__(self, key):
        # Ensure key is a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Fill missing dimensions with full slices
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))

        t_idx, z_idx, c_idx, y_idx, x_idx = key
        
        # Handle Z-Projection if z_idx is a slice
        if isinstance(z_idx, slice):
            # Extract the subset: (1, Z_subset, C_subset, Y_subset, X_subset)
            # We need to be careful with dimensions.
            # Let's just slice the array normally first.
            # self.array is (T, Z, C, Y, X)
            
            # We want to project along axis 1 (Z)
            # But we only want to project if the caller asked for a specific T and C?
            # Vispy usually asks for: [t, z_slice, :, :, :]
            
            # Let's slice everything EXCEPT Z first? No, that's hard.
            
            # Let's do it simply:
            # 1. Slice
            subset = self.array[key]
            
            # 2. If Z was sliced, the result 'subset' will have a Z dimension.
            # However, the caller of this proxy (Vispy/Visuals) expects (C, Y, X) or (Y, X)
            # when it asks for a specific Z.
            # If it asks for a Z-slice, it probably expects the projection.
            
            # Wait, if we just return self.array[key], we get a 5D/4D/3D array with the Z-dimension preserved.
            # But our Visuals expect a 2D or 3D array (Y, X) or (C, Y, X) for a "slice".
            
            # If z_idx is a slice, we MUST project to collapse that dimension.
            
            # Check if the resulting array has the Z-dimension where we expect it.
            # If we sliced [t, z_slice, c, y, x], the result shape depends on which are ints vs slices.
            
            # Let's assume the standard usage: [t_int, z_slice, :, :, :] -> Result (Z', C, Y, X)
            # We want to max-project along axis 0 of that result?
            
            # Actually, let's look at how we construct the slice.
            # If we just use numpy slicing, we get the data.
            data = self.array[key]
            
            # Now we need to know which axis corresponds to Z in 'data' so we can max-project it.
            # In (T, Z, C, Y, X), Z is axis 1.
            # If T is int, Z is axis 0.
            
            axis_to_project = None
            
            # Count how many dimensions before Z were kept (i.e. were slices, not ints)
            # T is index 0.
            current_dim = 0
            if isinstance(t_idx, slice):
                # T is kept
                current_dim += 1
            
            # Z is next. Since z_idx IS a slice (we are in this block), it is kept.
            axis_to_project = current_dim
            
            # Perform projection
            if data.size > 0:
                return np.max(data, axis=axis_to_project)
            else:
                return data # Empty
                
        else:
            # Normal slicing
            return self.array[key]


def load_image(filepath, use_memmap=True):
    """
    Loads an image and normalizes it to (T, Z, C, Y, X).
    Returns: (image_data_proxy, metadata_dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    # --- IMARIS PATH ---
    if ext == ".ims":
        reader = ImarisReader(filepath)
        data = Imaris5DProxy(reader)

        meta = {
            "filename": os.path.basename(filepath),
            "shape": data.shape,
            "scale": reader.voxel_size,  # (Z, Y, X)
            "channels": reader.channels_info,
        }
        return data, meta

    # --- TIFF PATH ---
    # Use generic generic wrapper for consistency
    if use_memmap:
        img = tifffile.memmap(filepath)
    else:
        img = tifffile.imread(filepath)

    ndim = img.ndim
    final_img = img

    # Normalization to (T, Z, C, Y, X)
    if ndim == 2:  # (Y, X) -> (1, 1, 1, Y, X)
        final_img = img[np.newaxis, np.newaxis, np.newaxis, :, :]
    elif ndim == 3:  # Assume (Z, Y, X) -> (1, Z, 1, Y, X)
        final_img = img[np.newaxis, :, np.newaxis, :, :]
    elif ndim == 4:  # Assume (Z, C, Y, X) -> (1, Z, C, Y, X)
        final_img = img[np.newaxis, :, :, :, :]

    scale = (1.0, 1.0, 1.0)
    
    # Wrap in Proxy
    data_proxy = Numpy5DProxy(final_img)

    return data_proxy, {
        "filename": os.path.basename(filepath),
        "shape": final_img.shape,
        "scale": scale,
    }
