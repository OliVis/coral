import tensorflow as tf
import numpy as np
from interpeter import CoralInterpreter
import model

# Convert function into a TensorFlow graph
@tf.function
def func(a, b):
    return tf.matmul(a, b)

# Create a trace of the function directly
concrete_func = func.get_concrete_function(
    a = tf.TensorSpec(shape=(None,2,2), dtype=tf.float32),
    b = tf.TensorSpec(shape=(None,2,2), dtype=tf.float32)
)

# Creates a complex 2x2 tensor
def random_complex():
    r = 5 # Values between -r and r
    a = np.random.uniform(-r, r)
    b = np.random.uniform(-r, r)

    # Matrix representation of a+bi
    return np.array([[a, -b],
                     [b, a]], dtype=np.float32)

# Dataset used for quantization
def representative_dataset():
    for _ in range(500):
        #  Input tensors with each a complex number
        a = np.array([random_complex()], dtype=np.float32)
        b = np.array([random_complex()], dtype=np.float32)
        
        yield [a, b]

# Convert the TensorFlow model to TensorFlow Lite and compile for Coral Edge TPU
model.convert_and_compile("model.tflite", concrete_func, representative_dataset)

# Initialize CoralInterpreter with the compiled TensorFlow Lite model
interpreter = CoralInterpreter("model_edgetpu.tflite", [2,2,2])

# Set input tensors for the interpreter
# 3+4i, 3-4i
interpreter.inputs[0] = np.array([[[ 3, -4],
                                   [ 4,  3]],
                                  [[ 3,  4],
                                   [-4,  3]]], dtype=np.float32)
# 2-3i, 3+2i
interpreter.inputs[1] = np.array([[[ 2,  3],
                                   [-3,  2]],
                                  [[ 3, -2],
                                   [ 2, 3]]], dtype=np.float32)

# Run quantized inference
interpreter.quantize_input()
interpreter.invoke()
interpreter.dequantize_output()

# Display the output tensor
print("Output Tensors:")
print(interpreter.outputs[0])
#print(np.round(interpreter.outputs[0]))
