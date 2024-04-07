import tensorflow as tf
import numpy as np
import coral

# =============================================================================
#                            MODEL CREATION
# =============================================================================

# Convert function into a TensorFlow graph
@tf.function
def func(a, b):
    return tf.matmul(a, b)

# Create a trace of the function directly
concrete_func = func.get_concrete_function(
    a = tf.TensorSpec(shape=(None,2,2), dtype=tf.float32),
    b = tf.TensorSpec(shape=(None,2,2), dtype=tf.float32)
)

# Dataset used for quantization
def representative_dataset():
    for _ in range(500):
        #  Input tensors with each a complex number in matrix form
        a = np.array([coral.complex_to_matrix(coral.random_complex(5))], dtype=np.float32)
        b = np.array([coral.complex_to_matrix(coral.random_complex(5))], dtype=np.float32)
        yield [a, b]

# Convert the TensorFlow model to TensorFlow Lite and compile for Coral Edge TPU
coral.create_model("model.tflite", concrete_func, representative_dataset)

# =============================================================================
#                            MODEL INVOCATION
# =============================================================================

# A constant for the number of complex samples
NUM_SAMPLES = 10

# Initialize Interpreter with the TensorFlow Lite model and input size
interpreter = coral.EdgeTPUInterpreter("model_edgetpu.tflite", [NUM_SAMPLES,2,2])

# Generate random complex numbers
complex_a = coral.random_complex(5, NUM_SAMPLES).astype(np.complex64)
complex_b = coral.random_complex(5, NUM_SAMPLES).astype(np.complex64)

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
