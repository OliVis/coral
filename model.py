import tensorflow as tf
import subprocess

def convert_and_compile(model_name: str, concrete_function, representative_dataset) -> None:
    """
    Converts a TensorFlow model to a TensorFlow Lite model with quantization optimized for the Coral Edge TPU,
    and then compiles the TensorFlow Lite model for use with the Coral device.

    Args:
        model_name (str): The filename for the saved TensorFlow Lite model.
        concrete_function: A TensorFlow ConcreteFunction that represents the model.
        representative_dataset: A function that returns a representative dataset
                                      for model quantization.
    """
    # Initialize the TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])

    # Ensure the model uses integer operations for Edge TPU compatibility
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert the TensorFlow model to TensorFlow Lite
    tflite_model = converter.convert()
    print("Conversion to TFLite format completed.")

    # Save the converted TensorFlow Lite model
    with open(model_name, "wb") as file:
        file.write(tflite_model)
    print(f"Model saved to {model_name}.")

    # Compile the TensorFlow Lite model for the Coral Edge TPU. 
    # The '-s' flag generates a summary of the compilation.
    result = subprocess.run(["edgetpu_compiler", "-s", model_name])

    # Verify the compilation process. A non-zero return code indicates an error.
    if result.returncode != 0:
        raise RuntimeError("Compilation for Edge TPU failed.")
