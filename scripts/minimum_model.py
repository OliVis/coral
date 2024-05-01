import os
import math
from typing import Generator
import argparse
import subprocess
import tensorflow as tf

# Minimal TensorFlow model with the FFT's input/output structure for testing.
# Note that some of the comments are not accurate anymore.

class FFT:
    def __init__(self, size: int) -> None:
        """
        Initialize an FFT (Fast Fourier Transform) instance with a specified size.
        Use the `compute` method compute the FFT using the Cooley-Tukey algorithm.

        Args:
            size (int): The size of the FFT, which must be a power of 2.
        """
        # Properties of the FFT
        self.size = size                   # The size of the input

    @tf.function
    def compute(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Compute the Fast Fourier Transform (FFT) of the input tensor using the Cooley-Tukey algorithm.

        Args:
            tensor (tf.Tensor): Input tensor with shape (`self.size`, 2), where `self.size` is the number of
                                complex numbers, represented as pairs of floats (real, imaginary).

        Returns:
            tf.Tensor: Transformed tensor containing the FFT result with the same shape (`self.size`, 2).
        """
        return tensor + 1

    def concrete_function(self) -> tf.types.experimental.ConcreteFunction:
        """
        Create and return a concrete function for the model.

        Returns:
            tf.ConcreteFunction: The concrete function for the model.
        """
        # Create a trace of the function directly
        concrete_func = self.compute.get_concrete_function(
            tf.TensorSpec(shape=(self.size, 2), dtype=tf.float32)
        )
        return concrete_func

    def representative_dataset(self) -> Generator[tf.Tensor, None, None]:
        """
        Generate a representative dataset for quantization.

        Yields:
            Generator[tf.Tensor, None, None]: A generator yielding random tensors.
        """
        for _ in range(500):
            yield [tf.random.uniform(shape=[self.size, 2], minval=-1, maxval=1, dtype=tf.float32)]

    def export_model(self, model_name: str) -> None:
        """
        Build and compile a TensorFlow Lite model optimized for the Coral Edge TPU.

        Args:
            model_name (str): The name of the created model file.
        """
        # Initialize the TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_concrete_functions([self.concrete_function()])

        # Ensure the model uses integer operations for Edge TPU compatibility
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert the TensorFlow model to TensorFlow Lite
        tflite_model = converter.convert()
        print("Conversion to TFLite format completed.")

        # Append the TensorFlow Lite extension to the model name
        file_name = model_name + ".tflite"

        # Save the converted TensorFlow Lite model
        with open(file_name, "wb") as file:
            file.write(tflite_model)
        print(f"Model saved to {file_name}.")

        # Compile the TensorFlow Lite model for the Coral Edge TPU. 
        # The '-s' flag generates a summary of the compilation.
        result = subprocess.run(["edgetpu_compiler", "-s", file_name])

        # Verify the compilation process. A non-zero return code indicates an error.
        if result.returncode != 0:
            raise RuntimeError("Compilation for Edge TPU failed.")

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build and compile an FFT (Fast Fourier Transform) TensorFlow Lite model optimized for the Coral Edge TPU.")
    parser.add_argument("size", type=int, help="Input size of the FFT.")
    parser.add_argument("name", type=str, help="Name of the created TensorFlow Lite model.")
    parser.add_argument("--outdir", type=str, default=None, help="Location to save the output model. Default is current directory.")

    return parser.parse_args()

def main() -> None:
    """
    Build and compile an FFT (Fast Fourier Transform) TensorFlow Lite model optimized for the Coral Edge TPU.
    """
    args = parse_args()

    # Change to the specified output location if provided
    if args.outdir is not None:
        os.chdir(args.outdir)

    # Create a FFT instance and export the TensorFlow Lite model
    fft_model = FFT(args.size)
    fft_model.export_model(args.name)

if __name__ == "__main__":
    main()
