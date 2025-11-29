
import numpy as np
from impy.io import normalize_to_5d

def test_dims_normalization():
    # 1. Test Heuristics (No dims)
    # 2D (Y, X) -> (1, 1, 1, Y, X)
    arr_2d = np.zeros((100, 100))
    proxy = normalize_to_5d(arr_2d)
    print(f"2D -> {proxy.shape}")
    assert proxy.shape == (1, 1, 1, 100, 100)
    
    # 3D (Z, Y, X) -> (1, Z, 1, Y, X)
    arr_3d = np.zeros((10, 100, 100))
    proxy = normalize_to_5d(arr_3d)
    print(f"3D (Default) -> {proxy.shape}")
    assert proxy.shape == (1, 10, 1, 100, 100)
    
    # 2. Test Explicit Dims
    # 'tyx' -> (T, 1, 1, Y, X)
    proxy = normalize_to_5d(arr_3d, dims='tyx')
    print(f"3D ('tyx') -> {proxy.shape}")
    assert proxy.shape == (10, 1, 1, 100, 100)
    
    # 'cyx' -> (1, 1, C, Y, X)
    arr_cyx = np.zeros((3, 100, 100))
    proxy = normalize_to_5d(arr_cyx, dims='cyx')
    print(f"3D ('cyx') -> {proxy.shape}")
    assert proxy.shape == (1, 1, 3, 100, 100)
    
    # 'zcyx' -> (1, Z, C, Y, X)
    arr_4d = np.zeros((5, 2, 100, 100))
    proxy = normalize_to_5d(arr_4d, dims='zcyx')
    print(f"4D ('zcyx') -> {proxy.shape}")
    assert proxy.shape == (1, 5, 2, 100, 100)
    
    # 'tcyx' -> (T, 1, C, Y, X)
    proxy = normalize_to_5d(arr_4d, dims='tcyx')
    print(f"4D ('tcyx') -> {proxy.shape}")
    assert proxy.shape == (5, 1, 2, 100, 100)
    
    # 'czyx' -> (1, Z, C, Y, X) (Transpose needed)
    # Input: (C=2, Z=5, Y=100, X=100)
    arr_czyx = np.zeros((2, 5, 100, 100))
    proxy = normalize_to_5d(arr_czyx, dims='czyx')
    print(f"4D ('czyx') -> {proxy.shape}")
    assert proxy.shape == (1, 5, 2, 100, 100)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_dims_normalization()
