import numpy as np
from typing import Union

def complex_to_matrix(c: np.complexfloating) -> np.ndarray:
    """
    Converts a complex number 'c' into a 2x2 matrix representation.

    Args:
        c (np.complexfloating): Input complex number.

    Returns:
        np.ndarray: 2x2 matrix representation of the complex number.
    """
    # Create a 2x2 matrix from the real and imaginary parts of the complex number
    return np.array([[c.real, -c.imag],
                     [c.imag,  c.real]])

def matrix_to_complex(m: np.ndarray) -> np.complexfloating:
    """
    Converts a 2x2 matrix 'm' back to a complex number.

    Args:
        m (np.ndarray): Input 2x2 matrix.

    Returns:
        np.complexfloating: Complex number representation of the matrix.
    """
    # Check if the input matrix has the correct shape (2x2)
    if m.shape != (2, 2):
        raise ValueError("Input matrix must be of shape (2, 2)")

    # Extract elements from the matrix to form a complex number
    # The real part is m[0, 0] and the imaginary part is m[1, 0]
    # Create a complex number using these elements
    return m[0, 0] + m[1, 0] * 1j

def random_complex(radius: int, num_samples: int = None) -> Union[np.complexfloating, np.ndarray]:
    """
    Generate random complex numbers with specified radius.

    Args:
        radius (int): The maximum absolute value for the real and imaginary parts.
        num_samples (int, optional): Number of random complex numbers to generate. 
                                     If None, a single random complex number is generated.
                                     Defaults to None.

    Returns:
        Union[np.complexfloating, np.ndarray]: Random complex number or array of random complex numbers.
    """
    # Generate random real and imaginary parts within the specified radius
    real_part = np.random.uniform(-radius, radius, num_samples)
    imag_part = np.random.uniform(-radius, radius, num_samples) * 1j

    # Combine real and imaginary parts to form complex numbers
    return real_part + imag_part
