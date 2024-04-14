import numpy as np
from fft_model import FFT

FFT_SIZE = 8

def main():
    """
    Tests the FFT model's accuracy, TensorFlow Lite conversion, and Edge TPU compatibility.
    It compares the model's FFT calculations with those from NumPy's FFT.
    """
    fft = FFT(FFT_SIZE)

    # Test model conversion to TensorFlow Lite and Edge TPU compilation
    fft.export_model("test")

    # Generate random values for the real and imaginary parts of the input complex numbers
    real_part = np.random.uniform(-127, 127, FFT_SIZE).astype(np.float32)
    imag_part = np.random.uniform(-127, 127, FFT_SIZE).astype(np.float32)

    # Combine the real and imaginary parts to form a complex array
    complex_input = real_part + imag_part * 1j

    # Prepare the input for the FFT model as a stacked array of real and imaginary components
    # Real parts as first row, imaginary parts as second row
    model_input = np.stack([real_part, imag_part])

    # Calculate the FFT using NumPy for comparison
    numpy_output = np.fft.fft(complex_input)
    
    # Calculate the FFT using the TensorFlow model
    model_output = fft.compute(model_input)

    # Print the results for comparison
    print(f"Input:\n{complex_input}")
    print(f"NumPy:\n{numpy_output}")
    print(f"Model:\n{model_output}")

if __name__ == "__main__":
    main()
