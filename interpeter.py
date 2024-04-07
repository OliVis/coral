from pycoral.utils import edgetpu
import numpy as np

class CoralInterpreter:
    """
    Facilitates TensorFlow Lite model inference on Coral Edge TPU devices,
    handling the entire inference cycle including input quantization, model execution,
    and output dequantization.
    """

    def __init__(self, model_name: str, input_size: list[int]) -> None:
        """
        Initializes a CoralInterpreter instance with the specified Edge TPU model.

        Args:
            model_name (str): The file path to the TensorFlow Lite model compiled for the Edge TPU.
            input_size (list[int]): The shape/size of input tensors expected by the model.
        """
        # Initialize the Edge TPU interpreter with the specified model
        self._interpreter = edgetpu.make_interpreter(model_name)

        # Retrieve details about the model's input and output tensors
        self.input_details = self._interpreter.get_input_details()
        self.output_details = self._interpreter.get_output_details()

        # Number of inputs and outputs
        self.n_inputs = len(self.input_details)
        self.n_outputs = len(self.output_details)

        # Placeholder for input and output tensors
        self.inputs = [None] * self.n_inputs
        self.outputs = [None] * self.n_outputs

        # Resize input tensors based on the provided input_size
        for i in range(self.n_inputs):    
            self._interpreter.resize_tensor_input(self.input_details[i]["index"], input_size)

        # Allocate memory for input and output tensors
        self._interpreter.allocate_tensors()

    def quantize_input(self) -> None:
        """
        Quantizes the input data for all inputs stored in `self.inputs` according to each input tensor's
        quantization parameters. It's expected that `self.inputs` is populated with float32 numpy arrays.
        """
        for i, data in enumerate(self.inputs):
            if data is None:
                raise ValueError(f"Input at index {i} has not been set.")
            scale, zero_point = self.input_details[i]["quantization"]
            self.inputs[i] = np.clip(np.round(data / scale + zero_point), -127, 127).astype(np.int8)

    def invoke(self) -> None:
        """
        Performs inference using the current (quantized) inputs, updating `self.outputs` with the raw results.
        """
        # if None in self.inputs:
        #     raise RuntimeError("Model invocation with unset inputs.")

        # Set the input tensors in the interpreter
        for i, tensor in enumerate(self.inputs):
            self._interpreter.set_tensor(self.input_details[i]["index"], tensor)

        # Run inference
        self._interpreter.invoke()

        # Retrieve and store the raw output tensors from the interpreter
        for i in range(self.n_outputs):
            self.outputs[i] = self._interpreter.get_tensor(self.output_details[i]['index'])

    def dequantize_output(self) -> None:
        """
        Dequantizes the data for all outputs stored in `self.outputs` according to each output tensor's
        quantization parameters.
        """
        for i, data in enumerate(self.outputs):
            if data is None:
                raise ValueError(f"Output at index {i} has not been retrieved from the interpreter.")
            scale, zero_point = self.output_details[i]["quantization"]
            self.outputs[i] = (data.astype(np.float32) - zero_point) * scale
