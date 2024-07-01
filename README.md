# Signal Processing with Coral Edge TPU 

This project uses the Coral Edge TPU and TensorFlow Lite to perform signal processing functions, specifically the Fast Fourier Transform (FFT), on the Coral Edge TPU device. By leveraging the power of the Edge TPU, we aim to accelerate these computations for real-time or high-performance signal processing tasks.

## Getting Started

### Prerequisites

Ensure that you have the following installed on your system:

- Python with TensorFlow 2.16.1
- CMake 3.18+
- gasket-dkms
- librtlsdr-dev

### Installation

To install this project, follow these steps:

1. Download and install the Edge TPU runtime library `libedgetpu1-std` and `libedgetpu-dev`. You can obtain them from [here](https://github.com/feranick/libedgetpu/releases/tag/16.0TF2.16.1-1) or you can build `libedgetpu` from [source](https://github.com/google-coral/libedgetpu). Make sure to choose the appropriate package version for your device. Note that Google's [packages](https://coral.ai/docs/m2/get-started/#2-install-the-pcie-driver-and-edge-tpu-runtime) are outdated and won't work with this project.

2. Clone the project repository:
```bash
git clone https://github.com/OliVis/coral.git
cd coral
```

3. Create a build directory and run CMake to configure the project. Make sure the TensorFlow version matches the installed Python TensorFlow version.
```bash
mkdir build
cd build
cmake ..
```

4. Build the executable using make:
```bash
make -j$(nproc)
```
Use the `-j$(nproc)` option to use multiple processing units and speed up the build process.

## Usage
```bash
./coral -f <fft_size> -s <samples> -o <output_file> [-n output_samples>] [-m <model_script>]
```
- `fft_size`: The size of the FFT, must be a power of 2.
- `samples`: The number of samples to batch process.
- `output_file`: The binary file to output the results to.
- `output_samples`: (optional) The number of samples in the output file. If not provided, the program will keep running indefinitely.
- `model_script`: (optional) The Python script used for model creation. Default is `fft_model.py`.

`2 * fft_size * samples` bytes are read from the USB per callback.
That number must be a multiple of 256 and is recommended to be a multiple of 16384 (USB urb size) for optimal performance.
