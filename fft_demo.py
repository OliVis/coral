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

# Generate an array of random complex numbers for input
complex_input = coral.random_complex(COMPLEX_RADIUS, NUM_SAMPLES).astype(np.complex64)

# Split the real and imaginary parts to create the model input
model_input = np.array([complex_input.real, complex_input.imag])

# Calculate the expected output using NumPy's FFT
numpy_output = np.fft.fft(complex_input)

# Use the model to compute the output directly without quantization
model_output = fft.compute(model_input).numpy()
model_output = model_output[0] + model_output[1] * 1j # Convert back to complex numbers

# Set the model input to the interpreter
interpreter.inputs[0] = model_input

# Run the quantized inference on the Edge TPU
interpreter.run_inference()

# Retrieve the output from the interpreter and convert it back to complex numbers
coral_output = interpreter.outputs[0][0] + interpreter.outputs[0][1] * 1j

# Print the results for comparison
print(f"Input:\n{complex_input}")
print(f"NumPy:\n{numpy_output}")
print(f"Model:\n{model_output}")
print(f"Coral:\n{coral_output}")
