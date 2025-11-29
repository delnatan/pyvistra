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

            # Stack into (C, ...)
            stack = np.array(planes)

            # If Z was also sliced, stack is (C, Z, Y, X).
            # We want (Z, C, Y, X).
            if isinstance(z, slice):
                stack = np.transpose(stack, (1, 0, 2, 3))

            # Apply Y and X slicing to the stack
            # Note: stack is (C, Y, X) or (Z, C, Y, X)
            # If (Z, C, Y, X), we apply y_idx to dim 2, x_idx to dim 3
            if stack.ndim == 4:
                return stack[:, :, y_idx, x_idx]
            else:
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
            # Z-Stack
            start, stop, step = z.indices(self.shape[1])
            z_indices = range(start, stop, step)
            
            if len(z_indices) == 0:
                return np.zeros((0, self.shape[3], self.shape[4]), dtype=self.dtype)

            # Read all planes in range
            stack = []
            for z_i in z_indices:
                stack.append(self.reader.read(c=c, t=t, z=z_i))
            
            # Return stack (Z, Y, X)
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
