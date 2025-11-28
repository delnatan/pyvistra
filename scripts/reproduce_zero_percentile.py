import numpy as np

def test_percentile():
    print("--- Test 1: No Zeros ---")
    # Random data between 100 and 200
    data = np.random.uniform(100, 200, (1000, 1000))
    mn, mx = np.nanpercentile(data, (0.1, 99.9))
    print(f"Min (0.1%): {mn:.2f}")
    print(f"Max (99.9%): {mx:.2f}")
    
    print("\n--- Test 2: With Zeros (Background) ---")
    # 10% zeros
    mask = np.random.random((1000, 1000)) < 0.1
    data[mask] = 0
    mn, mx = np.nanpercentile(data, (0.1, 99.9))
    print(f"Min (0.1%): {mn:.2f}")
    print(f"Max (99.9%): {mx:.2f}")
    
    print("\n--- Test 3: With Zeros (Masked) ---")
    # Ignore zeros
    valid_data = data[data > 0]
    mn, mx = np.nanpercentile(valid_data, (0.1, 99.9))
    print(f"Min (0.1%): {mn:.2f}")
    print(f"Max (99.9%): {mx:.2f}")

if __name__ == "__main__":
    test_percentile()
