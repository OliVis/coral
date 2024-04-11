import time
import threading
import numpy as np
from rtlsdr import RtlSdr

def callback(values: np.ndarray, context) -> None:
    """
    Callback function that handles received samples.

    Args:
        values (np.ndarray): Array of complex samples received from RTL-SDR.
        context: Arbitrary object passed as an argument to the callback function.
    """
    print(values)

def main() -> None:
    # Initialize RTL-SDR device
    sdr = RtlSdr()

    # Configure device settings
    sdr.sample_rate = 2.048e6 # Hz
    sdr.center_freq = 70e6    # Hz
    sdr.gain = 'auto'

    # Arguments for asynchronous sampling
    function_kwargs = {
        "callback": callback, # Callback function that will be called on every data receive
        "num_samples": 8192,  # Number of samples to read per asynchronous call (16KB URB size)
        "context": None       # Object to be passed as an argument to the callback.
    }

    # Start a new thread for asynchronous sampling
    threading.Thread(target=sdr.read_samples_async, kwargs=function_kwargs, daemon=True).start()

    # Continue sampling for 10 seconds
    time.sleep(10)

if __name__  == "__main__":
    main()
