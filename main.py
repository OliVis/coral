import tensorflow as tf
import numpy as np
from interpeter import CoralInterpreter
import model

# Global properties for the TensorFlow model
type = tf.float32
shape = (2,2)

# Convert function into TensorFlow graph
@tf.function
def func(a, b):
    return tf.matmul(a, b)

# Create a trace of the function directly
concrete_func = func.get_concrete_function(
    a = tf.TensorSpec(shape=shape, dtype=type),
    b = tf.TensorSpec(shape=shape, dtype=type)
)

# Dataset used for quantization
def representative_dataset():
    for _ in range(500):
        a = tf.random.uniform(shape, -5, 5, dtype=type)
        b = tf.random.uniform(shape, -5, 5, dtype=type)
        yield [a, b]

# Convert the TensorFlow model to TensorFlow Lite and compile for Coral Edge TPU
model.convert_and_compile("model.tflite", concrete_func, representative_dataset)

# Initialize CoralInterpreter with the compiled TensorFlow Lite model
interpreter = CoralInterpreter("model_edgetpu.tflite")

# Set input tensors for the interpreter
interpreter.inputs[0] = np.array([[1, 2], [3, 4]], dtype=np.float32)
interpreter.inputs[1] = np.array([[1, 2], [3, 4]], dtype=np.float32)

# Run quantized inference
interpreter.quantize_input()
interpreter.invoke()
interpreter.dequantize_output()

# Display the output tensor
print("Output Tensor:")
print(interpreter.outputs[0])