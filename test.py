import numpy as np
from fft_model import FFT

FFT_SIZE = 8

def main():
    """
    Tests the FFT model's accuracy, TensorFlow Lite conversion, and Edge TPU compatibility.
    It compares the model's FFT calculations with those from NumPy's FFT.
    """
    # Initialize the FFT model
    fft = FFT(FFT_SIZE)

    # Test model conversion to TensorFlow Lite and Edge TPU compilation
    fft.export_model("test")

    # Generate a 2D array representing an array complex numbers in the format produced by the rtl-sdr
    # Each complex number consists of a pair of floats: [real, imag, real, imag, ...]
    data_array = np.random.uniform(low=-1, high=1, size=(FFT_SIZE, 2)).astype(np.float32)

    # Calculate the FFT using the TensorFlow model
    model_output = fft.compute(data_array)

    # Create complex numbers from pairs of floats
    complex_array = data_array[:, 0] + 1j * data_array[:, 1]

    # Calculate the FFT using NumPy for comparison
    numpy_output = np.fft.fft(complex_array)

    # Print the results for comparison
    print(f"Input:\n{data_array}")
    print(f"NumPy:\n{numpy_output}")
    print(f"Model:\n{model_output}")

if __name__ == "__main__":
    main()
