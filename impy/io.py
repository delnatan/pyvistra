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
        """
        # Ensure key is a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Fill missing dimensions with full slices
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))

        t_idx, z_idx, c_idx, y_idx, x_idx = key

        # --- Handle Time Slicing ---
        if isinstance(t_idx, slice):
            # Iterate over timepoints
            start, stop, step = t_idx.indices(self.shape[0])
            t_indices = range(start, stop, step)
            
            if len(t_indices) == 0:
                # Return empty array with correct dimensionality
                # We need to know the shape of the rest to return correct empty
                # Let's just return empty of 5D?
                # Shape: (0, Z', C', Y', X')
                # It's complex to calculate exact shape without reading.
                # Simplified: return empty array
                return np.empty((0,) + self.shape[1:], dtype=self.dtype)

            stack = []
            for t in t_indices:
                stack.append(self._read_timepoint(t, z_idx, c_idx))
            
            # Stack along Time (axis 0)
            # Result: (T, ...)
            data = np.array(stack)
            
            # Apply Y/X slicing
            # data is (T, Z, C, Y, X) or (T, C, Y, X) etc.
            # We need to apply y_idx, x_idx to the last two dimensions
            return data[..., y_idx, x_idx]
            
        else:
            # Single Timepoint
            data = self._read_timepoint(t_idx, z_idx, c_idx)
            return data[..., y_idx, x_idx]

    def _read_timepoint(self, t, z_idx, c_idx):
        """
        Reads a single timepoint with Z and C slicing.
        Returns data with shape (Z, C, Y, X) or subset.
        """
        # --- Handle Channel Slicing ---
        if isinstance(c_idx, slice):
            start, stop, step = c_idx.indices(self.shape[2])
            channels = range(start, stop, step)
            
            planes = []
            for c in channels:
                planes.append(self._read_z_slice(c, t, z_idx))

            # Stack into (C, ...)
            stack = np.array(planes)

            # If Z was also sliced (or is full stack), stack is (C, Z, Y, X).
            # We want (Z, C, Y, X).
            # If z_idx was int, stack is (C, Y, X) -> No transpose needed.
            if stack.ndim == 4:
                stack = np.transpose(stack, (1, 0, 2, 3))
            
            return stack

        else:
            # Single channel
            return self._read_z_slice(c_idx, t, z_idx)

    def _read_z_slice(self, c, t, z):
        """
        Helper to read Z-slice/stack for specific C and T.
        Optimized to use full-volume read if z is full slice.
        """
        if isinstance(z, slice):
            start, stop, step = z.indices(self.shape[1])
            z_indices = range(start, stop, step)
            
            # Optimization: If full Z-stack requested (step=1 and full range)
            if step == 1 and start == 0 and stop == self.shape[1]:
                 return self.reader.read(c=c, t=t, z=None)

            if len(z_indices) == 0:
                return np.zeros((0, self.shape[3], self.shape[4]), dtype=self.dtype)

            # Read specific planes
            stack = []
            for z_i in z_indices:
                stack.append(self.reader.read(c=c, t=t, z=z_i))
            
            return np.array(stack)
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
        
        # Standard slicing
        return self.array[key]



def normalize_to_5d(data, dims=None):
    """
    Normalizes a numpy array to (T, Z, C, Y, X) format.
    
    Args:
        data (np.ndarray): Input array.
        dims (str): Optional dimension string (e.g. 'tyx', 'zcyx').
                    If None, heuristics are used.
                    
    Returns:
        Numpy5DProxy: Wrapped data.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy array")

    final_img = data
    
    if dims:
        dims = dims.lower()
        if len(dims) != data.ndim:
            raise ValueError(f"dims string length ({len(dims)}) must match data ndim ({data.ndim})")
            
        # Target: t, z, c, y, x
        target_order = ['t', 'z', 'c', 'y', 'x']
        
        present_dims = [d for d in target_order if d in dims]
        perm = [dims.index(d) for d in present_dims]
        
        final_img = np.transpose(data, perm)
        
        # Calculate target shape
        target_shape = []
        for char in target_order:
            if char in dims:
                target_shape.append(data.shape[dims.index(char)])
            else:
                target_shape.append(1)
        
        final_img = final_img.reshape(target_shape)

    else:
        # Heuristics
        ndim = data.ndim
        if ndim == 2:  # (Y, X) -> (1, 1, 1, Y, X)
            final_img = data[np.newaxis, np.newaxis, np.newaxis, :, :]
        elif ndim == 3:  # Assume (Z, Y, X) -> (1, Z, 1, Y, X)
            final_img = data[np.newaxis, :, np.newaxis, :, :]
        elif ndim == 4:  # Assume (Z, C, Y, X) -> (1, Z, C, Y, X)
            final_img = data[np.newaxis, :, :, :, :]
        elif ndim == 5:  # Assume (T, Z, C, Y, X)
            final_img = data

    return Numpy5DProxy(final_img)


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
    elif ndim == 5:  # Assume (T, Z, C, Y, X)
        final_img = img

    scale = (1.0, 1.0, 1.0)
    
    # Wrap in Proxy
    data_proxy = Numpy5DProxy(final_img)

    return data_proxy, {
        "filename": os.path.basename(filepath),
        "shape": final_img.shape,
        "scale": scale,
    }
