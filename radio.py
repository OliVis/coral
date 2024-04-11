import os
import threading
import numpy as np
from queue import Queue
from time import perf_counter
from rtlsdr import RtlSdr
import coral

# Input size of the FFT
FFT_SIZE = 1024

def callback(values: np.ndarray, context) -> None:
    """
    Callback function that handles received samples.

    Args:
        values (np.ndarray): Array of complex samples received from RTL-SDR.
        context: Arbitrary object passed as an argument to the callback function.
    """
    q = context # Retrieve the queue object from the context

    # Extract real and imaginary parts from the complex samples
    complex_array = np.array([values.real, values.imag])

    # Place the sample array into the queue
    q.put(complex_array)

def main() -> None:
    # Create a queue to hold received complex samples
    q = Queue()

    # ---------------------------- Coral Edge TPU ---------------------------- #
    # Determine the filename for the FFT model based on the FFT size
    model_name = coral.fft_model_name(FFT_SIZE)

    # Check if the model file already exists
    if os.path.exists(model_name):
        print("Using existing FFT model.")
    else:
        # Build the FFT model if it doesn't exist
        coral.build_fft_model(FFT_SIZE)

    # Initialize Interpreter with the TensorFlow Lite model and input size
    interpreter = coral.EdgeTPUInterpreterSingle(model_name, [2, FFT_SIZE])

    # ------------------------------- RTL-SDR -------------------------------- #
    # Initialize RTL-SDR device
    sdr = RtlSdr()

    # Configure device settings
    sdr.sample_rate = 2.048e6 # Hz
    sdr.center_freq = 70e6    # Hz
    sdr.gain = "auto"

    # Arguments for asynchronous sampling
    function_kwargs = {
        "callback": callback, # Callback function that will be called on every data receive
        "num_samples": 8192,  # Number of samples to read per asynchronous call (16KB URB size)
        "context": q          # Pass Queue Object to the callback function.
    }

    # Start a new thread for asynchronous sampling
    threading.Thread(target=sdr.read_samples_async, kwargs=function_kwargs, daemon=True).start()

    # --------------------------- Processing Loop ---------------------------- #
    # Record the start time to measure elapsed time later
    start_time = perf_counter()

    # Check if the queue has accumulated too many samples
    while q.qsize() < 1000:
        # Fetch the next batch of samples from the queue
        samples = q.get()

        # Split the samples into smaller arrays for processing
        # Assuming each sample batch is 8192 samples and FFT_SIZE is 1024
        for sample in np.split(samples, 8, axis=1):  # Each split has FFT_SIZE samples

            # Load the samples into the interpreter's input list
            interpreter.input = sample

            # Run the quantized inference on the Edge TPU
            interpreter.run_inference()

            # Print the model output
            print(interpreter.output)

            # Retrieve the output from the interpreter and convert it back to complex numbers
            # output = interpreter.outputs[0][0] + interpreter.outputs[0][1] * 1j

    # Calculate elapsed time since start
    elapsed_time = perf_counter() - start_time
    print("Runaway queue size detected. Samples are not being processed quickly enough.")
    print(f"Elapsed time: {elapsed_time}")

    # Shutdown gracefully
    sdr.cancel_read_async()
    sdr.close()

if __name__  == "__main__":
    main()
