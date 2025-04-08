# Signal Processing with Coral Edge TPU 

This repository is part of a research project that aims to use the Coral Edge TPU and TensorFlow Lite to perform signal processing functions, specifically the Fast Fourier Transform (FFT), on the Coral Edge TPU device. By leveraging the power of the Edge TPU, we aim to accelerate these computations for real-time or high-performance signal processing tasks.

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
./coral -s <fft_size> -b <batch_size> -o <output_file> [options]
```
Required arguments:
- `-s <fft_size>`       Size of the FFT (must be a power of 2).
- `-b <batch_size>`     Number of FFTs to compute per batch.*
- `-o <output_file>`    File to store the processed samples.

Optional arguments:
- `-i <iterations>`     Number of batches to process (default: run indefinitely).
- `-r <sample_rate>`    Sample rate of the SDR in Hz (default: 2,040,000Hz).
- `-f <frequency>`      Center frequency of the SDR in Hz (default: 80,000,000Hz).
- `-g <gain>`           Gain value of the SDR (default: automatic).
- `-d <samples_file>`   File to store the raw SDR samples.
- `-m <model_script>`   Python script used for model creation (default: 'fft_model.py').

*`2 * fft_size * batch_size` bytes are read from the SDR per callback. This must be a multiple of 512, and it's recommended to use multiples of 16,384 (USB URB size).

### Testing

```bash
./test.py {accuracy,performance} [coral_args]
```
Specify the type of test you want to perform. Use either:
- `accuracy`    Compare the accuracy of the Coral's FFT implementation with NumPy's FFT.
- `performance` Measure the processing times and count the number of dropped samples during execution.

Any additional arguments are passed to the Coral program. Notably:
- It is not necessary to provide an output file `-o` or a samples file `-d` as arguments, these will be handled automatically by the script.
- Limiting the number of iterations `-i` is required to control the test duration.

#### Additional notes
- Check out the `gpu` branch to run the identical program on the GPU rather than the Edge TPU, to compare hardware.
- Use `run_tests.sh` to automatically perform tests on a range of FFT sizes.
