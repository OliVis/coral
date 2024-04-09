import tensorflow as tf
import numpy as np
import coral

COMPLEX_RADIUS = 20 # Radius used for generating random complex numbers
NUM_SAMPLES = 8     # Number of complex samples and input size of the FFT

# =============================================================================
#                            MODEL CREATION
# =============================================================================

# Create an instance of FFT with the specified number of samples.
fft = coral.FFT(NUM_SAMPLES)

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
with tf.summary.create_file_writer("logs").as_default():
    tf.summary.trace_export(
        name="fft",
        step=0,
        profiler_outdir="logs"
    )

# =============================================================================
#                            MODEL INVOCATION
# =============================================================================

# Initialize Interpreter with the TensorFlow Lite model and input size
interpreter = coral.EdgeTPUInterpreter("model_edgetpu.tflite", [2,NUM_SAMPLES])

# Generate an array of random complex numbers
input_complex = coral.random_complex(COMPLEX_RADIUS, NUM_SAMPLES).astype(np.complex64)

# Prepare input for the interpreter by splitting real and imaginary parts
interpreter.inputs[0] = np.array([input_complex.real, input_complex.imag])

# Run the quantized inference on the Edge TPU
interpreter.run_inference()

# Calculate expected output using NumPy's FFT
expected_output = np.fft.fft(input_complex)

# Retrieve and convert the output from the interpreter back to complex numbers
model_output = interpreter.outputs[0][0] + interpreter.outputs[0][1] * 1j

# Print input, expected output, and model output for comparison
print(f"Input:\n{input_complex}")
print(f"Expected:\n{expected_output}")
print(f"Output:\n{model_output}")
