import tensorflow as tf
import numpy as np
import coral
import math

COMPLEX_RADIUS = 20 # Radius used for generating random complex numbers
NUM_SAMPLES = 8     # Number of complex samples

# =============================================================================
#                            MODEL CREATION
# =============================================================================

class FFT:
    def __init__(self, size: int) -> None:
        """
        Initialize an FFT (Fast Fourier Transform) instance with a specified size.

        Args:
            size (int): The size of the FFT, which must be a power of 2.

        To compute the Fast Fourier Transform (FFT) of an input tensor, use the `compute` method.
        The `compute` method applies the Cooley-Tukey FFT algorithm to the input tensor.
        """
        self.size = size
        self.stages = int(math.log2(size))

        # Create test twiddle factors for each stage
        self.twiddles = [tf.ones([2, 2**k]) for k in range(self.stages)]

    def twiddles():
        pass

    def butterfly(self, tensor: tf.Tensor, stage: int) -> tf.Tensor:
        """
        Perform a butterfly operation on a tensor for Fast Fourier Transform (FFT).

        Args:
            tensor (tf.Tensor): A 3D TensorFlow tensor with the shape (2, A, B).
            stage (int): The current FFT stage, used to select the appropriate twiddle factors.

        Returns:
            tf.Tensor: A TensorFlow tensor of the same shape as `tensor`, with the butterfly operation applied.

        The input tensor is expected to have a shape of (2, A, B), where:
        - The first dimension of size 2 represents the real and imaginary parts of complex numbers.
        - 'A' is the number of butterfly operations to perform in parallel.
        - 'B' is the input size of each butterfly operation.

        Visualization of the operation: https://tinyurl.com/radix2-butterfly
        where:
        - 'a' represents the "even" part of the input.
        - 'b' represents the "odd" part of the input.
        - 'Wn' represents the twiddle factors.
        """
        # Split the input tensor into its 'even' (a) and 'odd' (b) components
        evens, odds = tf.split(tensor, 2, axis=2)

        # Apply 'twiddle factors' (Wn) to 'odds' (b) using complex multiplication:
        # (a + bi) * (c + di), where 'odds' corresponds to (a, b) and 'self.twiddles[stage]' to (c, d)
        # Info: [0] = real component, [1] = imaginary component and twiddles are applied using broadcasting
        odds = tf.stack([ 
            # Real component: (a * c) − (b * d)
            odds[0] * self.twiddles[stage][0] - odds[1] * self.twiddles[stage][1],
            # Imag component: (a * d) + (b * c) 
            odds[0] * self.twiddles[stage][1] + odds[1] * self.twiddles[stage][0]  
        ])

        # Accumulate 'even' and transformed 'odd' components
        # Combine them into the final output tensor
        return tf.concat([evens + odds, evens - odds], axis=2)

    @tf.function
    def compute(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Compute the Fast Fourier Transform (FFT) of the input tensor using the Cooley-Tukey algorithm.

        Args:
            tensor (tf.Tensor): Input tensor with shape (2, self.size),
            where the first dimension of size 2 represents the real and imaginary parts of complex numbers.

        Returns:
            tf.Tensor: Transformed tensor representing the FFT result with shape (2, self.size).
        """
        for stage in range(self.stages):
            # Calculate butterfly size and count based on the current stage
            butterfly_size = 2 ** (stage + 1)
            butterfly_count = self.size // butterfly_size

            # Reshape the tensor to prepare for butterfly operations
            tensor = tf.reshape(tensor, [2, butterfly_count, butterfly_size])
            
            # Perform butterfly operations for the current stage
            tensor = self.butterfly(tensor, stage)

        # Reshape the tensor back to the original shape after all stages are processed
        return tf.reshape(tensor, [2, self.size])

# Create an instance of FFT with a specified number of samples.
fft = FFT(NUM_SAMPLES)

# Create a trace of the function directly
concrete_func = fft.compute.get_concrete_function(
    tf.TensorSpec(shape=(2, NUM_SAMPLES), dtype=tf.float32)
)

# Dataset used for quantization
def representative_dataset():
    for _ in range(500):
        yield [tf.random.uniform([2, NUM_SAMPLES], -COMPLEX_RADIUS, COMPLEX_RADIUS)]

# Convert the TensorFlow model to TensorFlow Lite and compile for Coral Edge TPU
coral.create_model("model.tflite", concrete_func, representative_dataset)

# =============================================================================
#                            MODEL VISUALIZATION
# =============================================================================

# Enable tracing
tf.summary.trace_on(graph=True, profiler=False)

# Call the function with the dummy input to trace the graph
fft.compute(tf.ones([2,NUM_SAMPLES]))

# Export the trace for visualization in TensorBoard
with tf.summary.create_file_writer('logs').as_default():
    tf.summary.trace_export(
        name="fft",
        step=0,
        profiler_outdir='logs'
    )

# Exit without invoking the model
exit()

# =============================================================================
#                            MODEL INVOCATION
# =============================================================================

# Initialize Interpreter with the TensorFlow Lite model and input size
interpreter = coral.EdgeTPUInterpreter("model_edgetpu.tflite", [NUM_SAMPLES,2,2])

# Generate random complex numbers
complex_a = coral.random_complex(COMPLEX_RADIUS, NUM_SAMPLES).astype(np.complex64)
complex_b = coral.random_complex(COMPLEX_RADIUS, NUM_SAMPLES).astype(np.complex64)

# Prepare inputs for the interpreter by converting complex numbers to matrices
interpreter.inputs[0] = np.array([coral.complex_to_matrix(n) for n in complex_a])
interpreter.inputs[1] = np.array([coral.complex_to_matrix(n) for n in complex_b])

# Run the quantized inference on the Edge TPU
interpreter.run_inference()

# Calculate multiplications of the complex numbers (for comparison)
expected_output = complex_a * complex_b

# Retrieve and convert the output from the interpreter back to complex numbers
model_output = np.array([coral.matrix_to_complex(m) for m in interpreter.outputs[0]])

# Print the comparison between expected output and model output for each sample
for i in range(NUM_SAMPLES):
    print(f"Sample {i + 1}:")
    print(f"Expected: {expected_output[i]}")
    print(f"Output:   {model_output[i]}")
    print()
