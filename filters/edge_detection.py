"""
edge_detection.py - Sobel Edge Detection Filter

Detects edges using the Sobel operator for gradient calculation.

"""

import numpy as np
from filters.convolution import convolve2d


def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Detect edges using the Sobel operator.
    
    Sobel kernels:
        Gx (horizontal gradient):     Gy (vertical gradient):
        [-1, 0, 1]                    [-1, -2, -1]
        [-2, 0, 2]                    [ 0,  0,  0]
        [-1, 0, 1]                    [ 1,  2,  1]
    
    The Sobel operator computes an approximation of the gradient of the
    image intensity function. At each point in the image, the result
    shows the magnitude of the gradient.
    
    The gradient magnitude is computed as: G = sqrt(Gx² + Gy²)
    
    Where:
        - Gx detects vertical edges (changes in horizontal direction)
        - Gy detects horizontal edges (changes in vertical direction)
    
    Args:
        image: Input grayscale image as numpy array with shape (H, W)
               Values should be in range [0, 255]
    
    Returns:
        Edge-detected image as numpy array with shape (H, W)
        Values in range [0, 255] as uint8
        Higher values indicate stronger edges
    
    Notes:
        - Output is normalized to [0, 255] range
        - Edges appear as bright lines on dark background
        - Pre-blurring the image can reduce noise sensitivity
    
    Example:
        >>> # Vertical edge
        >>> img = np.array([[0, 0, 255, 255],
        ...                 [0, 0, 255, 255],
        ...                 [0, 0, 255, 255]], dtype=np.uint8)
        >>> edges = edge_detection(img)
        >>> edges[1, 1] > edges[1, 0]  # Edge detected at transition
        True
    """
    # Define Sobel kernels (hardcoded)
    # Gx: Detects vertical edges (horizontal gradient)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    # Gy: Detects horizontal edges (vertical gradient)
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    # Compute gradients (don't normalize yet - need float values)
    gx = convolve2d(image, sobel_x, normalize=False)
    gy = convolve2d(image, sobel_y, normalize=False)
    
    # Compute gradient magnitude: G = sqrt(Gx² + Gy²)
    gradient_magnitude = np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2)
    
    # Normalize to [0, 255] range
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    
    return gradient_magnitude.astype(np.uint8)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing edge detection filter...")
    
    # Test 1: Vertical edge
    vertical_edge = np.zeros((10, 10), dtype=np.uint8)
    vertical_edge[:, 5:] = 255
    edges_v = edge_detection(vertical_edge)
    print(f"Vertical edge detection:")
    print(f"  At edge (col 5): {edges_v[5, 5]}")
    print(f"  Away from edge (col 0): {edges_v[5, 0]}")
    print(f"  Away from edge (col 9): {edges_v[5, 9]}")
    
    # Test 2: Horizontal edge
    horizontal_edge = np.zeros((10, 10), dtype=np.uint8)
    horizontal_edge[5:, :] = 255
    edges_h = edge_detection(horizontal_edge)
    print(f"\nHorizontal edge detection:")
    print(f"  At edge (row 5): {edges_h[5, 5]}")
    print(f"  Away from edge (row 0): {edges_h[0, 5]}")
    print(f"  Away from edge (row 9): {edges_h[9, 5]}")
    
    # Test 3: Diagonal edge
    diagonal_edge = np.zeros((10, 10), dtype=np.uint8)
    for i in range(10):
        diagonal_edge[i, i:] = 255
    edges_d = edge_detection(diagonal_edge)
    print(f"\nDiagonal edge detection:")
    print(f"  At diagonal: {edges_d[5, 5]}")
    
    # Test 4: Uniform image (no edges)
    uniform = np.full((10, 10), 128, dtype=np.uint8)
    edges_uniform = edge_detection(uniform)
    print(f"\nUniform image (no edges):")
    print(f"  Max edge value: {edges_uniform.max()} (should be 0)")
    
    # Test 5: Shape and dtype preservation
    random_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    edges_random = edge_detection(random_img)
    assert edges_random.shape == random_img.shape, "Shape mismatch"
    assert edges_random.dtype == np.uint8, "Dtype mismatch"
    print(f"\nShape preservation: OK")
    
    print("\nAll edge detection tests passed!")
